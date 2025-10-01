import torch
import torch.nn as nn
from einops import rearrange, repeat, einsum

class SelectiveSSM(nn.Module):
    """
    The Selective State-Space Model (SSM) block.
    """
    
    def __init__(self, d, N, s_A, use_delta, fix_sA, device):
        """
        Initialize the SSMBlock.
        Args:
            d (int): Number of input channels.
            N (int): Number of states per channel.
            s_A (float): Stability margin of matrix A.
            use_delta (bool): Whether to use the discretization parameter.
            fix_sA (bool): Whether to fix the first state of the first channel to -s_A.
            device (torch.device): Device to be used.
        Returns:
            None 
        """
        super(SelectiveSSM, self).__init__()

        self.d = d
        self.N = N
        self.s_A = s_A
        self.use_delta = use_delta
        self.fix_sA = fix_sA
        self.device = device

        # Initialize A as a dxN matrix where each row corresponds to the diagonal of an NxN matrix.
        if self.fix_sA:
            self.register_buffer('A', s_A*torch.ones(d, N, device=self.device))
        else:
            self.A = nn.Parameter(-10*torch.rand(d, N, device=self.device)+s_A)
            with torch.no_grad(): self.A[:,0] = s_A

        # Initialize W_B and W_C as projection weights for the input and output.
        self.W_B = nn.Parameter(torch.randn(N, d, device=self.device))
        self.W_C = nn.Parameter(torch.randn(N, d, device=self.device))

        # Initialize delta for the discretization parameter.
        self.q_delta = nn.Parameter(torch.randn(d, device=self.device))
        self.p_delta = nn.Parameter(torch.randn(1, device=self.device))
        
    def selective_scan(self, u, delta, A, B, C):
        """
        Perform the selective scan operation.
        Args:
            u (torch.Tensor): Input tensor of shape (batch_size, seq_len, d).
            delta (torch.Tensor): Discretization parameter of shape (batch_size, seq_len).
            A (torch.Tensor): Matrix A of shape (d, N) which is the diaginal of the large (Nd, Nd) matrix.
            B (torch.Tensor): Matrix B of shape (batch_size, seq_len, N).
            C (torch.Tensor): Matrix C of shape (batch_size, seq_len, N).
        Returns:
            y (torch.Tensor): Output tensor of shape (batch_size, seq_len, 1). 
        """
        
        batch_size, seq_len, d = u.shape
        N = A.shape[1]

        ##### Fix the first state of the first channel to -s_A #####
        # if self.fix_sA:
        #     with torch.no_grad(): A[:,:] = -self.s_A

        # Discretize A
        self.deltaA = torch.exp(einsum(delta, A, 'b l, d n -> b l d n'))  # Shape: (batch_size, seq_len, d, N)
        # print("A: ", A)
        # Discretize B and compute B*u
        deltaB_u = einsum(delta, B, u, 'b l, b l n, b l d -> b l d n')  # Shape: (batch_size, seq_len, d, N)

        # Perform sequential state-space computation
        x = torch.zeros((batch_size, d, N), device=u.device)
        ys = []
        for t in range(seq_len):
            x = self.deltaA[:, t] * x + deltaB_u[:, t]
            y = einsum(x, C[:, t, :], 'b d n, b n -> b d')
            ys.append(y)

        y = torch.stack(ys, dim=1)  # Shape: (batch_size, seq_len, d)

        return y
    
    def clip_state_matrix(self, threshold=-1e-6):
        with torch.no_grad():
            self.A.clamp_(max=threshold)

    def forward(self, u):
        """
        Forward pass for the SSMBlock.
        Args:
            u (torch.Tensor): Input tensor of shape (batch_size, seq_len, d).
        Returns:
            y (torch.Tensor): Output of the SSMBlock.
        """
        
        batch_size, seq_len, d = u.shape

        # Compute the discretization parameter \Delta if use_delta is enabled
        if self.use_delta:
            delta = self.p_delta + einsum(u, self.q_delta, 'b l d, d -> b l')  # Shape: (batch_size, seq_len)
            delta = torch.log(1 + torch.exp(delta)) # apply the soft plus function
        else:
            delta = torch.ones((batch_size, seq_len), device=self.device)  # Default to ones

        # Compute B and C using W_B and W_C
        B = einsum(self.W_B, u, 'n d, b l d -> b l n')  # Shape: (batch_size, seq_len, N)
        C = einsum(self.W_C, u, 'n d, b l d -> b l n')  # Shape: (batch_size, seq_len, N)

        # Perform selective scan
        y = self.selective_scan(u, delta, self.A, B, C)  # Shape: (batch_size, seq_len, d)
        
        return y

class SSMBlock(nn.Module):
    """
    Class to create a SSM Block using the Selective SSM.
    - Can be used for tasks that don't require a tokenized input.
    """

    def __init__(self, d, N, s_A, output_dim, use_delta, fix_sA, device):
        """
        Initialize the SSMBlock.
        Args:
            d (int): Number of input channels.
            N (int): Number of states per channel.
            s_A (float): Stability margin of matrix A.
            output_dim (int): Dimension of the output.
            use_delta (bool): Whether to use the discretization parameter.
            fix_sA (bool): Whether to fix the first state of the first channel to -s_A.
            device (torch.device): Device to be used.
        Returns:
            None
        """
    
        super(SSMBlock, self).__init__()
        
        self.d = d
        self.N = N
        self.s_A = s_A
        self.output_dim = output_dim
        self.use_delta = use_delta
        self.device = device

        # The slective ssm block
        self.ssm = SelectiveSSM(d, N, s_A, use_delta, fix_sA, device)

        #Initializa output weight
        self.w = nn.Parameter(torch.randn(d, device=self.device))
        
    def forward(self, u):
        
        y = self.ssm(u)
        out = einsum(self.w, y[:, -1, :], 'd, b d -> b')

        return out
    
class SentimentModel(nn.Module):
    """
    Class for the Sentiment Model using the SSMBlock.
    - Used for tasks that require tokenization (mainly text data).
    """

    def __init__(self, vocab_size, embedding_size=50, N=100, s_A=0, use_delta=True, fix_sA=True, device=None):
        """
        Initialize the SentimentModel.
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_size (int): Size of the embedding.
            N (int): Number of states per channel.
            s_A (float): Stability margin of matrix A.
            use_delta (bool): Whether to use the discretization parameter.
            fix_sA (bool): Whether to fix the first state of the first channel to -s_A.
            device (torch.device): Device to be used.
        Returns:
            None
        """
        
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.SSM = SSMBlock(d=embedding_size, N=N, s_A=s_A, output_dim=embedding_size, use_delta=use_delta, fix_sA=fix_sA, device=device)

    def forward(self, u):
        """
        Forward pass for the SentimentModel.
        Args:
            u (torch.Tensor): Input tensor of shape (batch_size, seq_len).
        Returns:
            y (torch.Tensor): Output of the SentimentModel
        """
        
        x = self.embedding(u)
        y = self.SSM(x)

        return y
    
class MultiClassModel(nn.Module):
    """
    Class for the MultiClassModel using the SSMBlock.
    - Used for tasks that output multiple class labels.
    """
    
    def __init__(self, vocab_size, embedding_size=50, N=100, s_A=0, num_classes=10, use_delta=True, fix_sA=True, device=None):
        """
        Initialize the MultiClassModel.
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_size (int): Size of the embedding.
            N (int): Number of states per channel.
            s_A (float): Stability margin of matrix A.
            num_classes (int): Number of classes.
            use_delta (bool): Whether to use the discretization parameter.
            fix_sA (bool): Whether to fix the first state of the first channel to -s_A.
            device (torch.device): Device to be used.
        Returns:
            None
        """
        
        print("vocab_size: ", vocab_size)
        print("embedding_size: ", embedding_size)
        
        super(MultiClassModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.ssm = SelectiveSSM(d=embedding_size, N=N, s_A=s_A, use_delta=use_delta, fix_sA=fix_sA, device=device)
        # self.ssm0 = SelectiveSSM(d=embedding_size, N=N, s_A=s_A, use_delta=use_delta, fix_sA=fix_sA, device=device)
        self.W = nn.Parameter(torch.randn(num_classes, embedding_size, device=device))
        
    def forward(self, u):
        """
        Forward pass for the MultiClassModel.
        Args:
            u (torch.Tensor): Input tensor of shape (batch_size, seq_len).
        Returns:
            y (torch.Tensor): Output of the MultiClassModel.
        """

        x = self.embedding(u)
        # x = self.ssm0(x)
        # x = torch.sigmoid(x)
        y = self.ssm(x)
        out = einsum(self.W, y[:, -1, :], 'c d, b d -> b c')

        return out
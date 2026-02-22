import torch
import torch.nn as nn

class Helmholtz_Loss(nn.Module):
    def __init__(self, wavenumber_k, grid_spacing_l):
        super().__init__()
        self.k = wavenumber_k
        self.l = grid_spacing_l
        self.grid_size = 32

        l_m = [grid_spacing_l**(i+1) for i in range(7)]
        
        c1 = torch.tensor([
            [l_m[0],       l_m[1]/2,     l_m[2]/3,     l_m[3]/4],
            [l_m[1]/2,     l_m[2]/3,     l_m[3]/4,     l_m[4]/5],
            [l_m[2]/3,     l_m[3]/4,     l_m[4]/5,     l_m[5]/6],
            [l_m[3]/4,     l_m[4]/5,     l_m[5]/6,     l_m[6]/7]
        ], dtype=torch.float32)

        c2 = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 4*l_m[0], 6*l_m[1]],
            [0, 0, 6*l_m[1], 12*l_m[2]]
        ], dtype=torch.float32)
        
        c3 = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2*l_m[0],   l_m[1],     (2/3)*l_m[2], (1/2)*l_m[3]],
            [3*l_m[1],   2*l_m[2],   (3/2)*l_m[3], (6/5)*l_m[4]]
        ], dtype=torch.float32)

        m_mat = torch.tensor([
            [1,  0, 0, 0],         
            [0,  0, 1, 0],          
            [-3, 3, -2, -1],   
            [2,  -2, 1,  1]    
        ], dtype=torch.float32)

        self.register_buffer('c1', c1)
        self.register_buffer('c2', c2) 
        self.register_buffer('c3', c3)
        self.register_buffer('m', m_mat) 

    def forward(self, x):
        real_start_channel = 0
        real_end_channel = 4
        imag_start_channel = 4
        imag_end_channel = 8
        
        x_real = x[:, real_start_channel:real_end_channel, :, :]
        x_imag = x[:, imag_start_channel:imag_end_channel, :, :]
        
        loss_real = self._compute_helmholtz(x_real)
        loss_imag = self._compute_helmholtz(x_imag)

        return loss_real + loss_imag

    def _compute_helmholtz(self, x_part):
        B, C, H, W = x_part.shape
        
        BATCH_SLICE   = slice(None)
        CHANNEL_SLICE = slice(None)
        
        col_start_left = 0
        col_end_left   = W - 1 
        
        col_start_right = 1
        col_end_right   = W 
        
        row_start_top = 0
        row_end_top   = H - 1
        
        row_start_bot = 1
        row_end_bot   = H

        slice_top_rows    = slice(row_start_top, row_end_top)
        slice_bot_rows    = slice(row_start_bot, row_end_bot)
        slice_left_cols   = slice(col_start_left, col_end_left)
        slice_right_cols  = slice(col_start_right, col_end_right)
        
        x_tl = x_part[BATCH_SLICE, CHANNEL_SLICE, slice_top_rows, slice_left_cols]
        x_tr = x_part[BATCH_SLICE, CHANNEL_SLICE, slice_top_rows, slice_right_cols]
        x_bl = x_part[BATCH_SLICE, CHANNEL_SLICE, slice_bot_rows, slice_left_cols]
        x_br = x_part[BATCH_SLICE, CHANNEL_SLICE, slice_bot_rows, slice_right_cols]
        
        u_tl, ux_tl, uy_tl, uxy_tl = x_tl[:, 0], x_tl[:, 1], x_tl[:, 2], x_tl[:, 3]
        u_tr, ux_tr, uy_tr, uxy_tr = x_tr[:, 0], x_tr[:, 1], x_tr[:, 2], x_tr[:, 3]
        u_bl, ux_bl, uy_bl, uxy_bl = x_bl[:, 0], x_bl[:, 1], x_bl[:, 2], x_bl[:, 3]
        u_br, ux_br, uy_br, uxy_br = x_br[:, 0], x_br[:, 1], x_br[:, 2], x_br[:, 3]

        # CORRECTED Q MATRIX LAYOUT
        row_1 = torch.stack([u_tl, u_tr, uy_tl*self.l, uy_tr*self.l], dim=-1)
        row_2 = torch.stack([u_bl, u_br, uy_bl*self.l, uy_br*self.l], dim=-1)
        row_3 = torch.stack([ux_tl*self.l, ux_tr*self.l, uxy_tl*(self.l**2), uxy_tr*(self.l**2)], dim=-1)
        row_4 = torch.stack([ux_bl*self.l, ux_br*self.l, uxy_bl*(self.l**2), uxy_br*(self.l**2)], dim=-1)

        Q = torch.stack([row_1, row_2, row_3, row_4], dim=-2)

        m_expanded = self.m.view(1, 1, 1, 4, 4)
        m_q = torch.matmul(m_expanded, Q)
        a_matrix = torch.matmul(m_q, m_expanded.transpose(-1, -2))

        return self._solve_integral(a_matrix)

    def _solve_integral(self, A):
        At = A.transpose(-1, -2)
        
        # Helper function to compute the trace across the batch and spatial dimensions
        def batch_trace(matrix):
            return torch.diagonal(matrix, dim1=-2, dim2=-1).sum(dim=-1)

        # CORRECTED INTEGRAL EVALUATION USING TRACE
        # term1: Tr(A^T C_1 A C_2)
        term1_mat = torch.matmul(torch.matmul(torch.matmul(At, self.c1), A), self.c2)
        term1 = batch_trace(term1_mat)
        
        # term2: Tr(A^T C_2 A C_1)
        term2_mat = torch.matmul(torch.matmul(torch.matmul(At, self.c2), A), self.c1)
        term2 = batch_trace(term2_mat)
        
        # term3: k^4 * Tr(A^T C_1 A C_1)
        term3_mat = torch.matmul(torch.matmul(torch.matmul(At, self.c1), A), self.c1)
        term3 = (self.k**4) * batch_trace(term3_mat)
        
        # term4: 2 * Tr(A^T C_3 A C_3^T)
        term4_mat = torch.matmul(torch.matmul(torch.matmul(At, self.c3), A), self.c3.transpose(-1, -2))
        term4 = 2 * batch_trace(term4_mat)
        
        # term5: 2 * k^2 * Tr(A^T C_3 A C_1)
        term5_mat = torch.matmul(torch.matmul(torch.matmul(At, self.c3), A), self.c1)
        term5 = 2 * (self.k**2) * batch_trace(term5_mat)
        
        # term6: 2 * k^2 * Tr(A^T C_1 A C_3^T)
        term6_mat = torch.matmul(torch.matmul(torch.matmul(At, self.c1), A), self.c3.transpose(-1, -2))
        term6 = 2 * (self.k**2) * batch_trace(term6_mat)
        
        patch_integrals = term1 + term2 + term3 + term4 + term5 + term6
        
        area = ((self.grid_size - 1) * self.l)**2

        return torch.abs(torch.sum(patch_integrals) / area)
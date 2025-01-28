# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/models/mutual_self_attention.py
from typing import Any, Dict, Optional
import torch
from einops import rearrange
from src.models.attention import TemporalBasicTransformerBlock
from .attention import BasicTransformerBlock

def torch_dfs(model: torch.nn.Module):
    r"""
    Perform a depth-first search (DFS) on a PyTorch model to collect all its submodules.

    Args:
        model (torch.nn.Module): The PyTorch model to traverse.

    Returns:
        List[torch.nn.Module]: A list of all submodules in the model, including the model itself.
    """
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

def filter_matrices_by_size(matrix_list, reference_matrix):
    """
    Filter a list of matrices based on their size, keeping only those that match the size of the reference matrix.

    Args:
        matrix_list (List[torch.Tensor]): A list of matrices to filter.
        reference_matrix (torch.Tensor): The reference matrix whose size is used for filtering.

    Returns:
        List[torch.Tensor]: A list of matrices whose size matches the reference matrix.
    """
    ref_shape = reference_matrix.shape[-2:]
    return [matrix for matrix in matrix_list if matrix.shape[-2:] == ref_shape]

class ReferenceAttentionControl:
    def __init__(
        self,
        unet,
        mode = "write",
        do_classifier_free_guidance = False,
        attention_auto_machine_weight = float("inf"),
        gn_auto_machine_weight = 1.0,
        style_fidelity = 1.0,
        reference_attn = True,
        reference_adain = False,
        fusion_blocks = "midup",
        batch_size = 1,
    ) -> None:
        """
        Initialize the ReferenceAttentionControl module.

        Args:
            unet: The UNet model to control.
            mode (str): The mode of operation, either "read" or "write". Reference unet: write. Denoising Unet: read.
            do_classifier_free_guidance (bool): Whether to use classifier-free guidance.
            attention_auto_machine_weight (float): Weight for attention control.
            gn_auto_machine_weight (float): Weight for group norm control.
            style_fidelity (float): Controls the fidelity to the reference style.
            reference_attn (bool): Whether to use reference attention.
            reference_adain (bool): Whether to use reference AdaIN.
            fusion_blocks (str): Which blocks to fuse, either "midup" or "full".
            batch_size (int): The batch size.
        """
        # 10. Modify self attention and group norm
        self.unet = unet
        assert mode in ["read", "write"]
        assert fusion_blocks in ["midup", "full"]
        self.reference_attn = reference_attn
        self.reference_adain = reference_adain
        self.fusion_blocks = fusion_blocks
        self.register_reference_hooks(
            mode,
            do_classifier_free_guidance,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            reference_attn,
            reference_adain,
            fusion_blocks,
            batch_size = batch_size,
        )
        self.point_embedding=[]
        
    def register_reference_hooks(
        self,
        mode,
        do_classifier_free_guidance,
        attention_auto_machine_weight,
        gn_auto_machine_weight,
        style_fidelity,
        reference_attn,
        reference_adain,
        dtype=torch.float16,
        batch_size=1,
        num_images_per_prompt=1,
        device=torch.device("cpu"),
        fusion_blocks="midup",
    ):
        """
        Register hooks for reference attention control.

        Args:
            mode (str): The mode of operation, either "read" or "write".
            do_classifier_free_guidance (bool): Whether to use classifier-free guidance.
            attention_auto_machine_weight (float): 注意力层的权重系数
            gn_auto_machine_weight (float): 归一化层的权重系数
            style_fidelity (float): 风格保真度参数，用于控制参考特征的融合强度。
            reference_attn (bool): Whether to use reference attention.
            reference_adain (bool): Whether to use reference AdaIN.
            dtype (torch.dtype): The data type for tensors.
            batch_size (int): The batch size.
            num_images_per_prompt (int): The number of images per prompt.
            device (torch.device): The device to use.
            fusion_blocks (str): Which blocks to fuse, either "midup" or "full".
        """
        MODE = mode
        do_classifier_free_guidance = do_classifier_free_guidance
        attention_auto_machine_weight = attention_auto_machine_weight
        gn_auto_machine_weight = gn_auto_machine_weight
        style_fidelity = style_fidelity
        reference_attn = reference_attn
        reference_adain = reference_adain
        fusion_blocks = fusion_blocks
        num_images_per_prompt = num_images_per_prompt
        dtype = dtype
        if do_classifier_free_guidance:
            uc_mask = (
                torch.Tensor(
                    [1] * batch_size * num_images_per_prompt * 16
                    + [0] * batch_size * num_images_per_prompt * 16
                )
                .to(device)
                .bool()
            )
        else:
            uc_mask = (
                torch.Tensor([0] * batch_size * num_images_per_prompt * 2)
                .to(device)
                .bool()
            )

        def hacked_basic_transformer_inner_forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            video_length=None,
        ):
            """
            Modified forward pass for the BasicTransformerBlock to support reference attention control.

            Args:
                hidden_states (torch.FloatTensor): The input hidden states.
                attention_mask (Optional[torch.FloatTensor]): The attention mask.
                encoder_hidden_states (Optional[torch.FloatTensor]): The encoder hidden states.
                encoder_attention_mask (Optional[torch.FloatTensor]): The encoder attention mask.
                timestep (Optional[torch.LongTensor]): The timestep for AdaIN.
                cross_attention_kwargs (Dict[str, Any]): Additional arguments for cross-attention.
                class_labels (Optional[torch.LongTensor]): The class labels for AdaIN.
                video_length: The length of the video (unused).

            Returns:
                torch.FloatTensor: The output hidden states.
            """
            if self.use_ada_layer_norm:  # False
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                (
                    norm_hidden_states,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                ) = self.norm1(
                    hidden_states,
                    timestep,
                    class_labels,
                    hidden_dtype=hidden_states.dtype,
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # 1. Self-Attention
            # self.only_cross_attention = False
            cross_attention_kwargs = (
                cross_attention_kwargs if cross_attention_kwargs is not None else {}
            )
            if self.only_cross_attention:
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states
                    if self.only_cross_attention
                    else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            else:
                #########################################################################
                if MODE == "write":
                    # NOTE: 在write模式下, 模型会将当前的归一化隐藏状态保存到bank中, 然后正常计算自注意力输出
                    self.bank.append(norm_hidden_states.clone())
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states
                        if self.only_cross_attention
                        else None,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                if MODE == "read":
                    # NOTE: 在read模式下, 从bank中读取特征, 将当前特征与bank中的特征融合
                    bank_fea = [
                        rearrange(
                            d.unsqueeze(1).repeat(1, 1, 1, 1),
                            "b t l c -> (b t) l c",
                        )
                        for d in self.bank
                    ]
                    try:
                        modify_norm_hidden_states = torch.cat(
                            [norm_hidden_states+self.point_bank_main[0].repeat(norm_hidden_states.shape[0],1,1)] + [bank_fea[0]+self.point_bank_ref[0].repeat(norm_hidden_states.shape[0],1,1)], dim=1
                        )
                        modify_norm_hidden_states_v = torch.cat(
                            [norm_hidden_states] + bank_fea, dim=1
                        )
                        # import ipdb;ipdb.set_trace()
                        hidden_states_uc = (
                        self.attn1(
                            norm_hidden_states + self.point_bank_main[0].repeat(norm_hidden_states.shape[0],1,1),
                            encoder_hidden_states = modify_norm_hidden_states,
                            encoder_hidden_states_v = modify_norm_hidden_states_v,
                            attention_mask = attention_mask,
                        )
                        + hidden_states
                        )
                    except:
                        modify_norm_hidden_states = torch.cat(
                            [norm_hidden_states] + bank_fea, dim=1
                        )
                        hidden_states_uc = (
                            self.attn1(
                                norm_hidden_states,
                                encoder_hidden_states=modify_norm_hidden_states,
                                attention_mask=attention_mask,
                            )
                            + hidden_states
                        )
                    if do_classifier_free_guidance:
                        hidden_states_c = hidden_states_uc.clone()
                        _uc_mask = uc_mask.clone()
                        if hidden_states.shape[0] != _uc_mask.shape[0]:
                            _uc_mask = (
                                torch.Tensor(
                                    [1] * (hidden_states.shape[0] // 3)
                                    + [0] * (hidden_states.shape[0] // 3)
                                    + [0] * (hidden_states.shape[0] // 3)
                                )
                                .to(device)
                                .bool()
                            )
                            _uc_mask_2 = (
                                torch.Tensor(
                                    [0] * (hidden_states.shape[0] // 3)
                                    + [1] * (hidden_states.shape[0] // 3)
                                    + [0] * (hidden_states.shape[0] // 3)
                                )
                                .to(device)
                                .bool()
                            )
                        hidden_states_c[_uc_mask] = (
                            self.attn1(
                                norm_hidden_states[_uc_mask],
                                encoder_hidden_states=norm_hidden_states[_uc_mask],
                                attention_mask=attention_mask,
                            )
                            + hidden_states[_uc_mask]
                        )
                        modify_norm_hidden_states = torch.cat(
                            [norm_hidden_states] + bank_fea, dim=1
                        )
                        hidden_states_c[_uc_mask_2] = (
                            self.attn1(
                                norm_hidden_states[_uc_mask_2],
                                encoder_hidden_states=modify_norm_hidden_states[_uc_mask_2],
                                attention_mask=attention_mask,
                            )
                            + hidden_states[_uc_mask_2]
                        )
                        hidden_states = hidden_states_c.clone()
                    else:
                        hidden_states = hidden_states_uc

                    # NOTE: 当前的模块只根据mode来修改self_attention, 不修改cross_attention.
                    if self.attn2 is not None:
                        # Cross-Attention
                        norm_hidden_states = (
                            self.norm2(hidden_states, timestep)
                            if self.use_ada_layer_norm
                            else self.norm2(hidden_states)
                        )
                        hidden_states = (
                            self.attn2(
                                norm_hidden_states,
                                encoder_hidden_states=encoder_hidden_states,
                                attention_mask=attention_mask,
                            )
                            + hidden_states
                        )

                    # Feed-forward
                    hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
                    return hidden_states
                ##########################################################################
            
            # import ipdb;ipdb.set_trace()
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            try:
                hidden_states = attn_output + hidden_states
            except:
                import ipdb;ipdb.set_trace()
            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep)
                    if self.use_ada_layer_norm
                    else self.norm2(hidden_states)
                )

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = (
                    norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                )

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states

        if self.reference_attn:
            ##############################################################################
            # NOTE: 根据传入的fusion_blocks的类型, 来获取attention_modules
            if self.fusion_blocks == "midup":
                attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock)
                ]
            attn_modules = sorted(
                attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            ###############################################################################

            for i, module in enumerate(attn_modules):
                # NOTE: 再修改模型的前向传播之前, 先将模型的原始的全向传播保存下来
                module._original_inner_forward = module.forward

                if isinstance(module, BasicTransformerBlock):
                    # NOTE: 替换模型的前向传播过程
                    module.forward = hacked_basic_transformer_inner_forward.__get__(
                        module, BasicTransformerBlock
                    )
                module.bank = []
                module.point_bank_ref=[]
                module.point_bank_main=[]
                module.attn_weight = float(i) / float(len(attn_modules))

    def update(self, writer, point_embedding_ref = None, point_embedding_main = None, dtype = torch.float16):
        """
        Update the reference attention control with features from a writer model.

        Args:
            writer: The writer model providing the features.
            point_embedding_ref: Reference point embeddings.
            point_embedding_main: Main point embeddings.
            dtype (torch.dtype): The data type for tensors.
        """
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, TemporalBasicTransformerBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in (
                        torch_dfs(writer.unet.mid_block)
                        + torch_dfs(writer.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in torch_dfs(writer.unet)
                    if isinstance(module, BasicTransformerBlock)
                ]
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            writer_attn_modules = sorted(
                writer_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            # import ipdb;ipdb.set_trace()
            for r, w in zip(reader_attn_modules, writer_attn_modules):
                r.bank = [v.clone().to(dtype) for v in w.bank]
                if point_embedding_main is not None:
                    r.point_bank_ref = filter_matrices_by_size(point_embedding_ref, r.bank[0])
                    r.point_bank_main = filter_matrices_by_size(point_embedding_main, r.bank[0])
                # w.bank.clear()

    def clear(self):
        """
        Clear the reference attention control features.
        """
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                    or isinstance(module, TemporalBasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock)
                    or isinstance(module, TemporalBasicTransformerBlock)
                ]
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            for r in reader_attn_modules:
                r.bank.clear()
                r.point_bank_ref.clear()
                r.point_bank_main.clear()

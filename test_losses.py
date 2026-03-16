"""Quick functional test for all loss functions."""
import torch
from ewad_cpdp_loss import EWADLoss, CPDPLoss, DualTeacherDistillationLoss, sparse_topk_to_dense, batch_sparse_to_dense
from config import EXPERIMENTS

V = 1000  # small vocab for test
B, T = 2, 5

# Dummy data
student_logits = torch.randn(B, T, V)
teacher_32b_lp = torch.log_softmax(torch.randn(B, T, V), dim=-1)
teacher_14b_lp = torch.log_softmax(torch.randn(B, T, V), dim=-1)
gold = torch.randint(0, V, (B, T))

# Test EWAD (full mode)
ewad = EWADLoss(vocab_size=V, mode='full')
loss, diag = ewad(student_logits, teacher_32b_lp, teacher_14b_lp, gold)
print(f"EWAD loss: {loss.item():.4f}, gate: {diag['gate_mean']:.4f}, agree: {diag['agreement_mean']:.4f}")
assert loss.item() >= 0, "EWAD loss should be non-negative"

# Test EWAD (confidence_only)
ewad_conf = EWADLoss(vocab_size=V, mode='confidence_only')
loss_c, diag_c = ewad_conf(student_logits, teacher_32b_lp, teacher_14b_lp, gold)
print(f"EWAD conf_only: {loss_c.item():.4f}, agree: {diag_c['agreement_mean']}")
assert diag_c['agreement_mean'] == -1.0, "Agreement should be -1 in confidence_only mode"

# Test EWAD (agreement_only)
ewad_agr = EWADLoss(vocab_size=V, mode='agreement_only')
loss_a, diag_a = ewad_agr(student_logits, teacher_32b_lp, teacher_14b_lp, gold)
print(f"EWAD agree_only: {loss_a.item():.4f}, w_32b: {diag_a['w_32b_mean']:.4f}")
assert abs(diag_a['w_32b_mean'] - 0.5) < 0.01, "Weights should be 0.5 in agreement_only mode"

# Test CPDP
cpdp = CPDPLoss()
loss_cp, diag_cp = cpdp(student_logits, teacher_32b_lp, teacher_14b_lp)
print(f"CPDP loss: {loss_cp.item():.4f}, kl_mutual: {diag_cp['kl_teacher_mutual_mean']:.4f}")
assert loss_cp.item() >= 0, "CPDP loss should be non-negative"

# Test all experiment configs via DualTeacherDistillationLoss
print("\nAll experiment configs:")
for name, cfg in EXPERIMENTS.items():
    dual = DualTeacherDistillationLoss(V, cfg)
    loss_d, diag_d = dual(student_logits, gold, teacher_32b_lp, teacher_14b_lp)
    print(f"  {name}: loss={loss_d.item():.4f}")
    assert not torch.isnan(loss_d), f"NaN loss for {name}"
    assert not torch.isinf(loss_d), f"Inf loss for {name}"

# Test sparse utilities
topk = [(3, -0.5), (7, -1.2), (42, -2.0)]
dense = sparse_topk_to_dense(topk, V)
prob_sum = dense.exp().sum().item()
print(f"\nSparse->Dense: sum(exp)={prob_sum:.6f} (should be ~1.0)")
assert abs(prob_sum - 1.0) < 1e-4, f"Probability sum should be ~1.0, got {prob_sum}"

batch_topk = [[[(1, -0.3), (2, -1.0)], [(5, -0.5)]], [[(3, -0.7)], [(8, -0.4), (9, -1.5)]]]
batch_dense = batch_sparse_to_dense(batch_topk, V)
row_sum = batch_dense[0, 0].exp().sum().item()
print(f"Batch sparse shape: {batch_dense.shape}, row sum: {row_sum:.6f}")
assert abs(row_sum - 1.0) < 1e-4, f"Batch row sum should be ~1.0, got {row_sum}"

# Test backward pass (gradients flow)
student_logits_grad = torch.randn(B, T, V, requires_grad=True)
dual_full = DualTeacherDistillationLoss(V, EXPERIMENTS['ewad_cpdp'])
loss_grad, _ = dual_full(student_logits_grad, gold, teacher_32b_lp, teacher_14b_lp)
loss_grad.backward()
assert student_logits_grad.grad is not None, "Gradients should flow through loss"
print(f"\nGradient check: grad norm = {student_logits_grad.grad.norm().item():.4f}")

print("\n✓ ALL TESTS PASSED")

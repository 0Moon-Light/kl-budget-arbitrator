import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import softmax

# ==========================================
# 1. Definicja symulowanego środowiska
# ==========================================
# Mamy 3 ekspertów i rozkład prawdziwy (True distribution)
x = np.linspace(-5, 5, 500)
dx = x[1] - x[0]

# Prawdziwy rozkład celu P_true(x) - mieszanka dwóch rozkładów Gaussa
p_true = 0.6 * norm.pdf(x, loc=-1.0, scale=0.8) + 0.4 * norm.pdf(x, loc=2.0, scale=0.6)
p_true /= np.sum(p_true * dx)

# Rozkłady 3 ekspertów:
p_experts = np.zeros((3, len(x)))
p_experts[0] = norm.pdf(x, loc=-0.8, scale=0.9)  # Ekspert 0: Dokładny, bliski celu
p_experts[1] = norm.pdf(x, loc=1.5, scale=1.2)   # Ekspert 1: Przesunięty (bias)
p_experts[2] = norm.pdf(x, loc=-3.0, scale=0.5)  # Ekspert 2: Outlier / Pewny siebie, błądzący

for i in range(3):
    p_experts[i] /= np.sum(p_experts[i] * dx)

# Konsensus grupy (średnia arytmetyczna ekspertów)
p_consensus = np.mean(p_experts, axis=0)
p_consensus /= np.sum(p_consensus * dx)

# ==========================================
# 2. Obliczenia wzorów z notatek (KL-Budget Expert Arbitrator)
# ==========================================
p_param = 0.3
L_i = np.zeros(3)  # Średnia strata
V_i = np.zeros(3)  # Wariancja straty
A_i = np.zeros(3)  # Zgodność (-D_KL od konsensusu)
eps = 1e-12

for i in range(3):
    log_loss_x = -np.log(p_experts[i] + eps)
    base_loss = np.sum(p_true * (p_experts[i] - p_true)**2 * dx)
    
    # 1. Średnia strata L_i(c)
    L_i[i] = (1 - p_param) * base_loss + p_param * np.sum(p_true * log_loss_x * dx)
    
    # 2. Wariancja straty V_i(c)
    V_i[i] = (1 - p_param) * base_loss + p_param * np.sum(p_true * (log_loss_x - L_i[i])**2 * dx)
    
    # 3. Zgodność A_i(c) = -D_KL(p_i || p_consensus)
    kl_div = np.sum(p_experts[i] * np.log((p_experts[i] + eps) / (p_consensus + eps)) * dx)
    A_i[i] = -kl_div

# ==========================================
# 3. Wagi Softmax
# ==========================================
def compute_weights(alpha, beta, gamma):
    logits = -alpha * L_i - beta * V_i + gamma * A_i
    return softmax(logits)

w_loss_only = compute_weights(alpha=2.0, beta=0.0, gamma=0.0)
w_low_var = compute_weights(alpha=1.0, beta=2.0, gamma=0.0)
w_kl_budget = compute_weights(alpha=1.0, beta=1.0, gamma=2.0)

# Ostateczne prognozy p(x_t) = sum_i w_i * p_i(x_t)
p_pred_loss = np.dot(w_loss_only, p_experts)
p_pred_kl = np.dot(w_kl_budget, p_experts)

# ==========================================
# 4. Wizualizacja Wyników
# ==========================================
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("KL-Budget Expert Arbitrator - Symulacja i Graficzna Weryfikacja", fontsize=15, fontweight='bold')

# Panel 1: Rozkłady Ekspertów
axs[0, 0].plot(x, p_true, 'k--', linewidth=2.5, label='Cel P_true(x)')
axs[0, 0].plot(x, p_experts[0], label='Ekspert 0 (Spójny & Celny)', color='#2ca02c')
axs[0, 0].plot(x, p_experts[1], label='Ekspert 1 (Bias)', color='#ff7f0e')
axs[0, 0].plot(x, p_experts[2], label='Ekspert 2 (Skrajny Outlier)', color='#d62728')
axs[0, 0].plot(x, p_consensus, ':', label='Konsensus Grupy', color='#7f7f7f')
axs[0, 0].set_title("1. Rozkłady Prawdopodobieństwa Ekspertów")
axs[0, 0].set_xlabel("x")
axs[0, 0].set_ylabel("Gęstość p(x)")
axs[0, 0].legend()
axs[0, 0].grid(True, alpha=0.3)

# Panel 2: Metryki (L_i, V_i, D_KL)
x_bar = np.arange(3)
width = 0.25
axs[0, 1].bar(x_bar - width, L_i, width, label='Strata L_i', color='#e377c2')
axs[0, 1].bar(x_bar, V_i, width, label='Wariancja V_i', color='#bcbd22')
axs[0, 1].bar(x_bar + width, -A_i, width, label='D_KL(p_i || p_consensus)', color='#17becf')
axs[0, 1].set_xticks(x_bar)
axs[0, 1].set_xticklabels(['Ekspert 0', 'Ekspert 1', 'Ekspert 2'])
axs[0, 1].set_title("2. Składowe arbitrażu (L_i, V_i, D_KL)")
axs[0, 1].legend()
axs[0, 1].grid(True, alpha=0.3)

# Panel 3: Wagi Softmax dla poszczególnych wariantów
x_bar2 = np.arange(3)
width2 = 0.25
axs[1, 0].bar(x_bar2 - width2, w_loss_only, width2, label='Tylko Strata (α=2, β=0, γ=0)', color='#1f77b4')
axs[1, 0].bar(x_bar2, w_low_var, width2, label='Strata + Wariancja (α=1, β=2, γ=0)', color='#ff7f0e')
axs[1, 0].bar(x_bar2 + width2, w_kl_budget, width2, label='KL-Budget (α=1, β=1, γ=2)', color='#2ca02c')
axs[1, 0].set_xticks(x_bar2)
axs[1, 0].set_xticklabels(['Ekspert 0', 'Ekspert 1', 'Ekspert 2'])
axs[1, 0].set_title("3. Dynamiczne Wagi Ekspertów W_i")
axs[1, 0].set_ylabel("Waga Softmax")
axs[1, 0].legend()
axs[1, 0].grid(True, alpha=0.3)

# Panel 4: Prognozy końcowe
axs[1, 1].plot(x, p_true, 'k--', linewidth=2.5, label='Cel P_true(x)')
axs[1, 1].plot(x, p_pred_loss, 'r-.', linewidth=2, label='Prognoza (Tylko Strata)')
axs[1, 1].plot(x, p_pred_kl, 'g-', linewidth=2, label='Prognoza (KL-Budget Arbitrator)')
axs[1, 1].set_title("4. Wynikowa Prognoza Komitetu p(x_t)")
axs[1, 1].set_xlabel("x")
axs[1, 1].set_ylabel("Gęstość p(x)")
axs[1, 1].legend()
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
out_file = "/home/moonlight/Pulpit/PHANTOMAI/kl_budget_arbitrator.png"
plt.savefig(out_file, dpi=300)
print(f"Wykres zapisany: {out_file}")

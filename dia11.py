"""
Resolução de Problema de Programação Linear

Problema:
Maximizar: Z = 200x1 + 300x2
Sujeito a:
    2x1 + x2 <= 20
    4x1 <= 32
    x2 <= 10
    x1, x2 >= 0

Solução:
    x1 = quantidade de produto A1 a ser produzido
    x2 = quantidade de produto A2 a ser produzido
"""
from scipy.optimize import linprog
import numpy as np
import matplotlib.pyplot as plt

# Função Objetivo: Maximizar Z = 200x1 + 300x2
funcao_objetivo = [-200, -300]

# Restrições de desigualdade (<=)
# 2x1 + x2 <= 20
# 4x1 + 0x2 <= 32
# 0x1 + x2 <= 10
matriz_restricoes = [
    [2, 1],   # 2x1 + x2 <= 20
    [4, 0],   # 4x1 <= 32
    [0, 1]    # x2 <= 10
]

limites_restricoes = [20, 32, 10]

# Limites das variáveis: x1 >= 0, x2 >= 0
limites_variaveis = [(0, None), (0, None)]

resultado = linprog(
    c=funcao_objetivo, # [-200, -300] para maximizar
    A_ub=matriz_restricoes, # [[2, 1], [4, 0]]
    b_ub=limites_restricoes, # [20, 10]
    bounds=limites_variaveis, # [(0, None), (0, None)]
    method='highs' # Método recomendado para problemas de programação linear
)

print("=" * 70)
print("SOLUÇÃO DO PROBLEMA DE PROGRAMAÇÃO LINEAR")
print("=" * 70)
print()

print("PROBLEMA:")
print("-" * 70)
print("Maximizar: Z = 200x1 + 300x2")
print()
print("Restrições:")
print("  2x1 + x2  <= 20")
print("  4x1       <= 32")
print("  x2        <= 10")
print("  x1, x2    >= 0")
print()

if resultado.success:
    x1, x2 = resultado.x
    z_maximo = -resultado.fun  # Negativo porque minimizamos o negativo
    
    print("SOLUÇÃO ÓTIMA ENCONTRADA:")
    print("-" * 70)
    print(f"Quantidade de Produto A1 (x1): {x1:.4f} unidades")
    print(f"Quantidade de Produto A2 (x2): {x2:.4f} unidades")
    print()
    print(f"Valor Máximo da Função Objetivo (Z): {z_maximo:.2f}")
    print()
    
    # Verificação das restrições
    print("VERIFICAÇÃO DAS RESTRIÇÕES:")
    print("-" * 70)
    restricao1 = 2*x1 + x2
    restricao2 = 4*x1
    restricao3 = x2
    
    print(f"Restrição 1: 2x1 + x2 = {restricao1:.4f} <= 20? {restricao1 <= 20 + 1e-6}")
    print(f"Restrição 2: 4x1 = {restricao2:.4f} <= 32? {restricao2 <= 32 + 1e-6}")
    print(f"Restrição 3: x2 = {restricao3:.4f} <= 10? {restricao3 <= 10 + 1e-6}")
    print(f"x1 >= 0? {x1 >= -1e-6}")
    print(f"x2 >= 0? {x2 >= -1e-6}")
    print()
    
    print("=" * 70)
    
else:
    print("ERRO: Não foi possível encontrar uma solução ótima!")
    print(f"Mensagem: {resultado.message}")

# ============================================================================
# VISUALIZAÇÃO GRÁFICA
# ============================================================================

if resultado.success:
    print()
    print("Gerando gráfico...")
    
    # Criar figura e eixos
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Intervalo de valores para x1 (numpy)
    x1_vals = np.linspace(0, 9, 400)
    
    # Equações das restrições usando numpy
    x2_rest1 = 20 - 2*x1_vals       # 2x1 + x2 = 20
    x1_rest2 = 8.0                   # 4x1 = 32
    x2_rest3 = np.full_like(x1_vals, 10.0)  # x2 = 10
    
    # Plotar as restrições
    ax.plot(x1_vals, x2_rest1, 'r-', linewidth=2, label='2x₁ + x₂ = 20')
    ax.axvline(x=x1_rest2, color='b', linewidth=2, label='4x₁ = 32')
    ax.axhline(y=10.0, color='purple', linewidth=2, label='x₂ = 10')
    
    # Preencher a região viável com numpy
    x2_region = np.minimum(x2_rest1, x2_rest3)
    x2_region = np.maximum(x2_region, 0)
    ax.fill_between(x1_vals, 0, x2_region, where=(x1_vals <= x1_rest2), alpha=0.3, color='green', label='Região Viável')
    
    # Plotar o ponto ótimo
    ax.plot(x1, x2, 'r*', markersize=20, label=f'Solução Ótima ({x1:.2f}, {x2:.2f})')
    
    # Curvas de nível da função objetivo usando numpy
    for z_val in [1000, 2000, 3000, 4000, 5000]:
        x2_obj = (z_val - 200*x1_vals) / 300
        ax.plot(x1_vals, x2_obj, '--', alpha=0.5, linewidth=1.5, label=f'Z = {z_val}')
    
    # Configurar eixos e labels
    ax.set_xlim(-0.5, 9)
    ax.set_ylim(-0.5, 15)
    ax.set_xlabel('x₁ (Produto A1)', fontsize=12, fontweight='bold')
    ax.set_ylabel('x₂ (Produto A2)', fontsize=12, fontweight='bold')
    ax.set_title('Problema de Programação Linear - Maximizar Z = 200x₁ + 300x₂', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    # Adicionar anotação no ponto ótimo
    ax.annotate(f'Z = {z_maximo:.2f}\n({x1:.2f}, {x2:.2f})',
                xy=(x1, x2), xytext=(x1+0.3, x2-2),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2))
    
    plt.tight_layout()
    plt.savefig('programacao_linear.png', dpi=150, bbox_inches='tight')
    print("✓ Gráfico salvo como 'programacao_linear.png'")
    plt.show()

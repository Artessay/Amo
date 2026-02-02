
import numpy as np
import matplotlib.pyplot as plt

def length_dependent_decay(L, L0):
    """
    Calculate the length-dependent decay coefficient R_len.
    
    Parameters:
    L (np.ndarray): Independent variable, representing length
    L0 (float): Characteristic length parameter
    
    Returns:
    np.ndarray: Decay coefficient values
    """
    return 1 / (1 + L / L0)

plt.style.use('default')
L0 = 384
L = np.linspace(0, 1024, 1024)
R_len = length_dependent_decay(L, L0)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(L, R_len, color='#1f77b4', linewidth=2, label=r'$R_{len} = \frac{1}{1 + \frac{L}{L_0}}$')

char_L = L0
char_R = 1 / (1 + char_L / L0)
ax.scatter(char_L, char_R, color='#ff7f0e', s=80, zorder=5)
ax.annotate(r'$L=L_0, R_{len}=0.5$',
            xy=(char_L, char_R),
            xytext=(char_L + 0.2, char_R + 0.1),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='#ff7f0e'))

ax.set_xlabel('Length $L$', fontsize=12)
ax.set_ylabel('Decay Coefficient $R_{len}$', fontsize=12)
ax.set_title('Inverse Length-Dependent Decay Coefficient', fontsize=14, fontweight='bold')

ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('vis_inverse_decay_curve.png', dpi=300, bbox_inches='tight')
plt.close()

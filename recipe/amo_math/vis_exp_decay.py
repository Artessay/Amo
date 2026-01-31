import numpy as np
import matplotlib.pyplot as plt

# Define the exponential decay function as given in the formula
def length_dependent_decay(L, L0):
    """
    Calculate the length-dependent decay coefficient R_len.
    
    Parameters:
    L (np.ndarray): Independent variable, representing length
    L0 (float): Characteristic length parameter
    
    Returns:
    np.ndarray: Decay coefficient values
    """
    return np.exp(-L / L0)

# Set plot style for better visualization
plt.style.use('default')

# Define characteristic length L0 (can be adjusted as needed)
L0 = 384

# Generate a sequence of length values from 0 to 5*L0 (covers the main decay trend)
L = np.linspace(0, 1024, 1024)

# Calculate decay coefficients
R_len = length_dependent_decay(L, L0)

# Create figure and axis objects
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the decay curve
ax.plot(L, R_len, color='#1f77b4', linewidth=2, label=r'$R_{len} = \exp\left(-\frac{L}{L_0}\right)$')

# Add characteristic point annotation (L=L0, R_len=1/e)
char_L = L0
char_R = np.exp(-1)
ax.scatter(char_L, char_R, color='#ff7f0e', s=80, zorder=5)
ax.annotate(r'$L=L_0, R_{len}=1/e$', 
            xy=(char_L, char_R), 
            xytext=(char_L + 0.2, char_R + 0.1),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='#ff7f0e'))

# Set axis labels and title
ax.set_xlabel('Length $L$', fontsize=12)
ax.set_ylabel('Decay Coefficient $R_{len}$', fontsize=12)
ax.set_title('Exponential Decay of Length-Dependent Coefficient', fontsize=14, fontweight='bold')

# Add grid and legend
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=10)

# Ensure tight layout to avoid label cutoff
plt.tight_layout()

# Save the figure to local path (modify the path as needed)
# Use high resolution (300 dpi) and avoid displaying the plot
plt.savefig('vis_exponential_decay_curve.png', dpi=300, bbox_inches='tight')
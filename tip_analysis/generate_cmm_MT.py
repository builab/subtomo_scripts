# Script to generate MT model
import math

def generate_microtubule_cmm(filename, protofilaments=13, repeats=8, radius=110, pitch=120, dimer_dist=80):
    """
    Generate a 13-protofilament microtubule in ChimeraX CMM format.
    Alpha tubulin (dark green) and beta tubulin (blueish) alternate in each protofilament.
    
    Parameters:
    filename - output CMM file name
    protofilaments - number of protofilaments (default 13)
    repeats - number of dimer repeats along the length (default 4)
    radius - microtubule radius in Å (default 125)
    pitch - helical pitch in Å (default 120)
    dimer_dist - distance between dimer centers in Å (default 80)
    """
    
    with open(filename, 'w') as f:
        f.write('<marker_set name="microtubule">\n')
        
        dimer_count = 0
        # Predefined colors (normalized 0-1)
        alpha_r, alpha_g, alpha_b = 0.13333, 0.5451, 0.1333  # dark green
        beta_r, beta_g, beta_b = 0.39216, 0.58431, 0.92941    # blueish
        radius_marker = "28"   # marker radius in Å
        
        # Microtubule parameters
        twist_per_dimer = math.radians(-360/protofilaments)  # left-handed helix
        rise_per_dimer = pitch/protofilaments
        
        for repeat in range(repeats):
            for pf in range(protofilaments):
                # Calculate position for alpha tubulin
                angle = -pf * 2 * math.pi / protofilaments
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                z = repeat * dimer_dist + pf * rise_per_dimer
                
                # Alpha tubulin (dark green)
                dimer_count += 1
                f.write(f'<marker id="{dimer_count}" x="{x:.4f}" y="{y:.4f}" z="{z:.4f}" '
                       f'r="{alpha_r:.4f}" g="{alpha_g:.4f}" b="{alpha_b:.4f}" '
                       f'radius="{radius_marker}"/>\n')
                
                # Beta tubulin (blueish) - offset along z-axis
                z_beta = z + 40  # 40Å offset between alpha and beta
                dimer_count += 1
                f.write(f'<marker id="{dimer_count}" x="{x:.4f}" y="{y:.4f}" z="{z_beta:.4f}" '
                       f'r="{beta_r:.4f}" g="{beta_g:.4f}" b="{beta_b:.4f}" '
                       f'radius="{radius_marker}"/>\n')
        
        f.write('</marker_set>\n')
    
    print(f"Generated {dimer_count} tubulin dimers in {filename}")
    print(f"Alpha tubulin color (RGB): {alpha_r:.4f}, {alpha_g:.4f}, {alpha_b:.4f}")
    print(f"Beta tubulin color (RGB): {beta_r:.4f}, {beta_g:.4f}, {beta_b:.4f}")

# Generate the microtubule
generate_microtubule_cmm("microtubule_13pf_colored.cmm")
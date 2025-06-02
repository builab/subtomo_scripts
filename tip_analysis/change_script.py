import argparse

def rgb_to_normalized(rgb_str):
    r, g, b = map(int, rgb_str.split(','))
    return f"{r/255:.3f},{g/255:.3f},{b/255:.3f}"

def modify_lines(lines, new_color, new_radius):
    updated_lines = []
    for line in lines:
        if '--color' in line:
            line = re.sub(r'--color\s+\d+\.?\d*,\d+\.?\d*,\d+\.?\d*', f'--color {new_color}', line)
        if '--r' in line:
            line = re.sub(r'--r\s+\d+', f'--r {new_radius}', line)
        updated_lines.append(line)
    return updated_lines

if __name__ == '__main__':
    import re

    parser = argparse.ArgumentParser(description='Update color and radius in a shell script.')
    parser.add_argument('--color', type=str, help='RGB color in 0–255 format, e.g. 250,250,250')
    parser.add_argument('--radius', type=int, help='Radius value, e.g. 8')
    parser.add_argument('--input', type=str, default='script.sh', help='Input .sh file')
    parser.add_argument('--output', type=str, default='script_modified.sh', help='Output .sh file')

    args = parser.parse_args()

    with open(args.input, 'r') as f:
        lines = f.readlines()

    new_color = rgb_to_normalized(args.color) if args.color else None
    new_radius = args.radius if args.radius is not None else None

    updated_lines = modify_lines(
        lines,
        new_color if new_color else '0.5,0.5,0.5',
        new_radius if new_radius else 8
    )

    with open(args.output, 'w') as f:
        f.writelines(updated_lines)

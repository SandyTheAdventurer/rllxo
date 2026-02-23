#!/usr/bin/env python3
"""
Circuit Visualization Tool
Reads a circuit JSON file and displays the circuit topology with component values
"""

import json
import sys
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch
import numpy as np

# PySpice is optional; waveform plotting is skipped if it's missing
try:
    from PySpice.Spice.Netlist import Circuit
    from PySpice.Unit import u_Hz, u_V
    PYSPICE_AVAILABLE = True
except Exception:
    PYSPICE_AVAILABLE = False

def load_circuit(filename):
    """Load circuit configuration from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def create_circuit_graph(circuit_data):
    """Create a NetworkX graph from circuit data"""
    G = nx.MultiGraph()
    
    # Add all nodes
    for node in circuit_data['nodes']:
        G.add_node(node)
    
    # Add edges for components
    edge_labels = {}
    for idx, comp in enumerate(circuit_data['components']):
        node1 = comp['node1']
        node2 = comp['node2']
        comp_type = comp['type']
        
        # Create label
        if 'value' in comp:
            value = comp['value']
            if comp_type == 'resistor':
                if value >= 1000:
                    label = f"R: {value/1000:.2f}kΩ"
                else:
                    label = f"R: {value:.2f}Ω"
            elif comp_type == 'capacitor':
                if value >= 1e-6:
                    label = f"C: {value*1e6:.2f}µF"
                elif value >= 1e-9:
                    label = f"C: {value*1e9:.2f}nF"
                else:
                    label = f"C: {value*1e12:.2f}pF"
            else:
                label = f"{comp_type[0].upper()}: {value}"
        else:
            label = comp_type[0].upper()  # Just first letter for diodes
        
        G.add_edge(node1, node2, label=label, comp_type=comp_type, key=idx)
        edge_labels[(node1, node2, idx)] = label
    
    return G, edge_labels


def simulate_waveforms(circuit_data, periods=3, load_resistance=1000.0):
    """Run a transient simulation and return time, input, and output waveforms."""
    if not PYSPICE_AVAILABLE:
        raise RuntimeError("PySpice not installed; install PySpice to view waveforms.")

    circuit = Circuit("CircuitViz")
    meta = circuit_data.get('meta', {})
    v_rms = float(meta.get('v_rms', 230.0))
    freq_hz = float(meta.get('freq_hz', 50.0))
    peak = v_rms * np.sqrt(2.0)

    circuit.SinusoidalVoltageSource(
        "source", "ac_p", "ac_n", amplitude=peak @ u_V, frequency=freq_hz @ u_Hz
    )
    circuit.V("V0", "ac_n", "0", 0 @ u_V)

    diode_count = 0
    resistor_count = 0
    cap_count = 0

    for comp in circuit_data['components']:
        ctype = comp['type']
        n1, n2 = comp['node1'], comp['node2']
        val = comp.get('value')
        if ctype == 'diode':
            diode_count += 1
            circuit.Diode(f"D{diode_count}", n1, n2, model="DGEN")
        elif ctype == 'resistor' and val is not None:
            resistor_count += 1
            circuit.R(f"R{resistor_count}", n1, n2, val)
        elif ctype == 'capacitor' and val is not None:
            cap_count += 1
            circuit.C(f"C{cap_count}", n1, n2, val)

    circuit.R("RLOAD", "out_p", "out_n", load_resistance)
    circuit.model("DGEN", "D", IS=4.352e-9, N=1.906, BV=110, IBV=0.0001, RS=0.6458)

    period = 1.0 / freq_hz
    step_time = period / 400.0
    end_time = max(periods, 1) * period

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(step_time=step_time, end_time=end_time)

    time_arr = np.array(analysis.time)
    vin = np.array(analysis['ac_p']) - np.array(analysis['ac_n'])
    vout = np.array(analysis['out_p']) - np.array(analysis['out_n'])
    return time_arr, vin, vout


def visualize_circuit(circuit_data, filename):
    """Visualize the circuit using NetworkX and Matplotlib"""
    G, edge_labels = create_circuit_graph(circuit_data)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2])

    # Main circuit visualization
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Try different layouts
    try:
        # Use spring layout with good spacing
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    except:
        pos = nx.circular_layout(G)
    
    # Draw nodes with different colors for special nodes
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if 'ac_' in node:
            node_colors.append('#ff6b6b')  # Red for AC input
            node_sizes.append(2000)
        elif 'out_' in node:
            node_colors.append('#51cf66')  # Green for output
            node_sizes.append(2000)
        else:
            node_colors.append('#74c0fc')  # Blue for internal nodes
            node_sizes.append(1500)
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                          alpha=0.9, ax=ax1)
    
    # Draw edges with different styles for different components
    edge_colors = []
    edge_styles = []
    edge_widths = []
    
    for u, v, key, data in G.edges(keys=True, data=True):
        comp_type = data.get('comp_type', 'unknown')
        if comp_type == 'diode':
            edge_colors.append('#fd7e14')
            edge_styles.append('solid')
            edge_widths.append(3)
        elif comp_type == 'resistor':
            edge_colors.append('#495057')
            edge_styles.append('solid')
            edge_widths.append(2)
        elif comp_type == 'capacitor':
            edge_colors.append('#7950f2')
            edge_styles.append('dashed')
            edge_widths.append(2)
        else:
            edge_colors.append('#adb5bd')
            edge_styles.append('solid')
            edge_widths.append(1)
    
    # Draw edges
    for idx, (u, v, key) in enumerate(G.edges(keys=True)):
        nx.draw_networkx_edges(G, pos, [(u, v)], width=edge_widths[idx],
                              edge_color=[edge_colors[idx]], style=edge_styles[idx],
                              ax=ax1, alpha=0.7)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax1)
    
    # Draw edge labels (component values)
    edge_label_dict = {}
    for (u, v, key), label in edge_labels.items():
        # For multigraph, we need to handle multiple edges between same nodes
        if (u, v) not in edge_label_dict:
            edge_label_dict[(u, v)] = label
        else:
            edge_label_dict[(u, v)] += f"\n{label}"
    
    nx.draw_networkx_edge_labels(G, pos, edge_label_dict, font_size=7, ax=ax1)
    
    ax1.set_title(f'Circuit Topology: {filename}', fontsize=14, fontweight='bold', pad=20)
    ax1.axis('off')
    
    # Circuit information panel
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    
    # Display circuit statistics and metadata
    info_text = []
    info_text.append("═" * 50)
    info_text.append("CIRCUIT INFORMATION")
    info_text.append("═" * 50)
    info_text.append("")
    
    # Metadata
    if 'meta' in circuit_data:
        meta = circuit_data['meta']
        info_text.append("📊 Metadata:")
        info_text.append(f"  • Source: {meta.get('source', 'N/A')}")
        if 'v_rms' in meta:
            info_text.append(f"  • Input Voltage (RMS): {meta['v_rms']:.1f} V")
            info_text.append(f"  • Input Voltage (Peak): {meta['v_rms'] * np.sqrt(2):.1f} V")
        if 'freq_hz' in meta:
            info_text.append(f"  • Frequency: {meta['freq_hz']:.1f} Hz")
        info_text.append("")
    
    # Component statistics
    info_text.append("🔧 Component Count:")
    component_types = {}
    for comp in circuit_data['components']:
        comp_type = comp['type']
        component_types[comp_type] = component_types.get(comp_type, 0) + 1
    
    for comp_type, count in sorted(component_types.items()):
        info_text.append(f"  • {comp_type.capitalize()}s: {count}")
    info_text.append(f"  • Total Components: {len(circuit_data['components'])}")
    info_text.append(f"  • Total Nodes: {len(circuit_data['nodes'])}")
    info_text.append("")
    
    # Node information
    info_text.append("📍 Special Nodes:")
    for node in sorted(circuit_data['nodes']):
        if 'ac_' in node:
            info_text.append(f"  • {node}: AC Input")
        elif 'out_' in node:
            info_text.append(f"  • {node}: Output")
    info_text.append("")
    
    # Component details
    info_text.append("📋 Component Details:")
    info_text.append("─" * 50)
    
    for idx, comp in enumerate(circuit_data['components'], 1):
        comp_type = comp['type']
        node1 = comp['node1']
        node2 = comp['node2']
        
        if 'value' in comp:
            value = comp['value']
            if comp_type == 'resistor':
                if value >= 1000:
                    value_str = f"{value/1000:.3f} kΩ"
                else:
                    value_str = f"{value:.3f} Ω"
            elif comp_type == 'capacitor':
                if value >= 1e-6:
                    value_str = f"{value*1e6:.3f} µF"
                elif value >= 1e-9:
                    value_str = f"{value*1e9:.3f} nF"
                else:
                    value_str = f"{value*1e12:.3f} pF"
            else:
                value_str = f"{value}"
            
            info_text.append(f"{idx:2d}. {comp_type.upper():12s} {node1:8s} → {node2:8s}  {value_str}")
        else:
            info_text.append(f"{idx:2d}. {comp_type.upper():12s} {node1:8s} → {node2:8s}")
    
    # Display text
    text_str = '\n'.join(info_text)
    ax2.text(0.05, 0.95, text_str, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff6b6b', 
                   markersize=10, label='AC Input'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#51cf66', 
                   markersize=10, label='Output'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#74c0fc', 
                   markersize=10, label='Internal Node'),
        plt.Line2D([0], [0], color='#fd7e14', linewidth=3, label='Diode'),
        plt.Line2D([0], [0], color='#495057', linewidth=2, label='Resistor'),
        plt.Line2D([0], [0], color='#7950f2', linewidth=2, linestyle='dashed', label='Capacitor'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=9)

    # Waveforms plot
    ax3 = fig.add_subplot(gs[1, :])
    try:
        time_arr, vin, vout = simulate_waveforms(circuit_data, periods=4)
        ax3.plot(time_arr, vin, label='Input (ac_p - ac_n)', color='#ff6b6b', linewidth=1.8)
        ax3.plot(time_arr, vout, label='Output (out_p - out_n)', color='#1c7ed6', linewidth=1.8)
        ax3.set_title('Input vs Output Voltage Over Time', fontsize=12, fontweight='bold', pad=12)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Voltage (V)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    except Exception as exc:
        ax3.axis('off')
        ax3.text(
            0.02,
            0.95,
            f"Waveform plot unavailable:\n{exc}",
            transform=ax3.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.4),
        )
    
    plt.tight_layout()
    
    # Save figure
    output_filename = filename.replace('.json', '_visualization.png')
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_filename}")
    
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 visualize_circuit.py <circuit_json_file>")
        print("Example: python3 visualize_circuit.py evolved_circuit.json")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        print(f"📖 Loading circuit from: {filename}")
        circuit_data = load_circuit(filename)
        
        print(f"🔍 Circuit loaded successfully!")
        print(f"   - Nodes: {len(circuit_data['nodes'])}")
        print(f"   - Components: {len(circuit_data['components'])}")
        
        print(f"🎨 Generating visualization...")
        visualize_circuit(circuit_data, filename)
        
    except FileNotFoundError:
        print(f"❌ Error: File '{filename}' not found!")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format in '{filename}'")
        print(f"   Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

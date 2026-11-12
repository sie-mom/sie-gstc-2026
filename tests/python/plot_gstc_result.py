import sys
import os
import matplotlib.pyplot as plt
import argparse

# Add applications directory to the path to import plot_field

from plot_field import plot_field_from_csv

def main():
    """
    Main function to plot 2D field data from a CSV file.
    """
    parser = argparse.ArgumentParser(
        description="Plot all 2D field components from the GSTC solver test."
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="jet",
        help=(
            "Matplotlib colormap name to use (e.g., viridis, RdBu_r). "
            "Use 'comsol' for the COMSOL-inspired diverging palette."
        ),
    )
    args = parser.parse_args()

    data_file = os.path.join(os.path.dirname(__file__), 'data', 'eh_total_2d_cut_gstc.csv')
    plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    components_to_plot = [
        'Ex_real', 'Ex_imag',
        'Ey_real', 'Ey_imag',
        'Ez_real', 'Ez_imag',
        'E_magnitude',
        'Hx_real', 'Hx_imag',
        'Hy_real', 'Hy_imag',
        'Hz_real', 'Hz_imag',
        'H_magnitude'
    ]

    for component in components_to_plot:
        print(f"Plotting component: {component}...")
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_title = f'{component} from {os.path.basename(data_file)}'

        # Set color limits: use fixed for real/imag, auto for magnitude
        vmin, vmax = None, None
        if 'magnitude' not in component and 'E' in component:
            vmin, vmax = -2.0, 2.0
        elif 'magnitude' not in component and 'H' in component:
            # H-field is E/eta, so its magnitude is much smaller.
            # eta0 is ~377, so H is ~1/377. Let's use a smaller range.
            vmin, vmax = -0.01, 0.01

        plot_field_from_csv(
            ax,
            data_file,
            component,
            plot_title,
            cmap=args.colormap,
            vmin=vmin,
            vmax=vmax
        )
        
        # --- Save the plot ---
        plt.tight_layout()

        # Define output filename and save the figure
        output_filename = os.path.join(plots_dir, f'{component}_2d_cut_gstc.png')
        print(f"  -> Saving plot to {output_filename}")
        plt.savefig(output_filename)
        plt.close(fig) # Close the figure to free up memory
    print("\nAll plots saved successfully.")

if __name__ == "__main__":
    main()
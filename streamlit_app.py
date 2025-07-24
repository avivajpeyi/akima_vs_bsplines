import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, splrep, splev, Akima1DInterpolator
import io
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

# Configure Streamlit page
st.set_page_config(
    page_title="B-spline vs Akima Comparison",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 300;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .method-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .instructions-box {
        background: #e3f2fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
    }
    .click-info {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    if 'data_points' not in st.session_state:
        st.session_state.data_points = []
    if 'plot_bounds' not in st.session_state:
        st.session_state.plot_bounds = {'xmin': 0, 'xmax': 10, 'ymin': -3, 'ymax': 3}


init_session_state()


def pixel_to_data_coords(x_pixel, y_pixel, img_width, img_height):
    """Convert pixel coordinates to data coordinates"""
    bounds = st.session_state.plot_bounds

    # Convert pixel coordinates to data coordinates
    x_data = bounds['xmin'] + (x_pixel / img_width) * (bounds['xmax'] - bounds['xmin'])
    y_data = bounds['ymax'] - (y_pixel / img_height) * (bounds['ymax'] - bounds['ymin'])

    return x_data, y_data


def add_point():
    """Callback function for adding/removing points on click"""
    if "plot_coords" in st.session_state:
        raw_value = st.session_state["plot_coords"]
        if raw_value is not None:
            x_pixel, y_pixel = raw_value["x"], raw_value["y"]

            # Get image dimensions (we'll use fixed size)
            img_width, img_height = 800, 600

            # Convert to data coordinates
            x_data, y_data = pixel_to_data_coords(x_pixel, y_pixel, img_width, img_height)

            # Check if click is near an existing point (for deletion)
            click_tolerance = 0.3  # Data coordinate tolerance
            for i, (px, py) in enumerate(st.session_state.data_points):
                if abs(px - x_data) < click_tolerance and abs(py - y_data) < click_tolerance:
                    # Remove the point
                    st.session_state.data_points.pop(i)
                    return

            # If not near existing point, add new point
            st.session_state.data_points.append((x_data, y_data))


def add_sample_data(data_type):
    """Add different types of sample data"""
    np.random.seed(42)  # For reproducible results

    if data_type == 'sine':
        x = np.linspace(0, 10, 12)
        y = np.sin(x * 0.8) + 0.1 * np.random.randn(len(x))
    elif data_type == 'noisy':
        x = np.linspace(0, 10, 15)
        y = np.sin(x) + 0.3 * np.random.randn(len(x)) + 0.2 * np.sin(3 * x)
    elif data_type == 'sawtooth':
        x = np.linspace(0, 10, 16)
        y = 2 * (x % 2) - 1 + 0.1 * np.random.randn(len(x))

    st.session_state.data_points = [(xi, yi) for xi, yi in zip(x, y)]


def clear_all_points():
    """Clear all data points"""
    st.session_state.data_points = []


def fit_bspline_custom(data_points, degree, smoothing):
    """Fit B-spline with automatic knot placement"""
    if len(data_points) < 2:
        return np.array([]), np.array([]), None

    try:
        # Extract x and y coordinates
        x_data = np.array([p[0] for p in data_points])
        y_data = np.array([p[1] for p in data_points])

        # Sort by x values
        sort_idx = np.argsort(x_data)
        x_data = x_data[sort_idx]
        y_data = y_data[sort_idx]

        # Use smoothing or automatic knots
        if smoothing > 0:
            tck = splrep(x_data, y_data, k=min(degree, len(x_data) - 1), s=smoothing)
        else:
            tck = splrep(x_data, y_data, k=min(degree, len(x_data) - 1))

        # Generate smooth curve
        x_fine = np.linspace(x_data.min(), x_data.max(), 200)
        y_fine = splev(x_fine, tck)

        return x_fine, y_fine, tck

    except Exception as e:
        st.error(f"B-spline fitting error: {e}")
        return np.array([]), np.array([]), None


def fit_akima(data_points):
    """Fit Akima spline"""
    if len(data_points) < 2:
        return np.array([]), np.array([])

    try:
        # Extract and sort coordinates
        x_data = np.array([p[0] for p in data_points])
        y_data = np.array([p[1] for p in data_points])

        sort_idx = np.argsort(x_data)
        x_data = x_data[sort_idx]
        y_data = y_data[sort_idx]

        akima = Akima1DInterpolator(x_data, y_data)
        x_fine = np.linspace(x_data.min(), x_data.max(), 200)
        y_fine = akima(x_fine)

        return x_fine, y_fine

    except Exception as e:
        st.error(f"Akima interpolation error: {e}")
        return np.array([]), np.array([])


def create_plot_image():
    """Create matplotlib plot and convert to PIL Image"""
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))

    # Set plot bounds
    bounds = st.session_state.plot_bounds
    ax.set_xlim(bounds['xmin'], bounds['xmax'])
    ax.set_ylim(bounds['ymin'], bounds['ymax'])

    # Grid and styling
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y', fontsize=14, fontweight='bold')
    ax.set_facecolor('#fafafa')

    # Data points
    if st.session_state.data_points:
        x_coords = [p[0] for p in st.session_state.data_points]
        y_coords = [p[1] for p in st.session_state.data_points]
        ax.scatter(x_coords, y_coords, s=120, c='black', marker='o',
                   edgecolors='white', linewidth=3, label='Data Points', zorder=5)

    # Fit splines if we have data
    if len(st.session_state.data_points) >= 2:
        degree = st.session_state.get('degree', 3)
        smoothing = st.session_state.get('smoothing', 0.0)

        # B-spline fit
        bx, by, tck = fit_bspline_custom(
            st.session_state.data_points,
            degree,
            smoothing
        )

        if len(bx) > 0:
            ax.plot(bx, by, color='#667eea', linewidth=5,
                    label='B-spline Fit', alpha=0.9)

        # Akima interpolation
        ax_data, ay_data = fit_akima(st.session_state.data_points)

        if len(ax_data) > 0:
            ax.plot(ax_data, ay_data, color='#e74c3c', linewidth=4,
                    linestyle=':', label='Akima Interpolation', alpha=0.9)

    # Title and instructions
    ax.set_title('B-spline vs Akima Spline Comparison\nClick to add points • Click existing points to delete',
                 fontsize=16, fontweight='bold', pad=20)

    # Legend
    if st.session_state.data_points:
        ax.legend(loc='upper right', frameon=True, fancybox=True,
                  shadow=True, fontsize=12, facecolor='white', framealpha=0.9)

    plt.tight_layout()

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close()

    # Convert to PIL Image
    img = Image.open(buf)

    # Resize to standard size for consistent coordinate mapping
    img = img.resize((800, 600), Image.Resampling.LANCZOS)

    return img


# Main app layout
st.markdown('<h1 class="main-header">B-spline vs Akima Spline Comparison</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Interactive tool for understanding Akima + Bspline fitting methods</p>',
    unsafe_allow_html=True)

# Instructions
st.markdown("""
<div class="instructions-box">
    <h3> How to Use</h3>
    <ol>
        <li><strong>Click on the plot:</strong> Add data points by clicking anywhere</li>
        <li><strong>Click existing points:</strong> Delete points by clicking near them</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("Controls")

# Data controls
st.sidebar.subheader("Sample Data")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Sine Wave"):
        add_sample_data('sine')
        st.rerun()
    if st.button("Sawtooth"):
        add_sample_data('sawtooth')
        st.rerun()

with col2:
    if st.button("Noisy Data"):
        add_sample_data('noisy')
        st.rerun()

if st.sidebar.button("🗑️ Clear All Points"):
    clear_all_points()
    st.rerun()

st.sidebar.markdown("---")

# B-spline parameters
st.sidebar.subheader("B-spline Parameters")

degree = st.sidebar.slider(
    "Spline Degree:",
    min_value=1,
    max_value=5,
    value=3,
    key='degree',
    help="Higher degrees create smoother curves but may oscillate more"
)

smoothing = st.sidebar.slider(
    "Smoothing Factor:",
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    step=0.1,
    key='smoothing',
    help="Higher values create smoother fits but may not follow data closely"
)

# Plot bounds controls
st.sidebar.markdown("---")
st.sidebar.subheader("Plot Range")
with st.sidebar.expander("Adjust Plot Bounds"):
    xmin = st.number_input("X min:", value=0.0, step=1.0)
    xmax = st.number_input("X max:", value=10.0, step=1.0)
    ymin = st.number_input("Y min:", value=-3.0, step=1.0)
    ymax = st.number_input("Y max:", value=3.0, step=1.0)

    if st.button("Update Bounds"):
        st.session_state.plot_bounds = {
            'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax
        }
        st.rerun()

st.sidebar.markdown("---")

# Display current point counts
st.sidebar.markdown("**Current Points:**")
st.sidebar.write(f"Data points: {len(st.session_state.data_points)}")

# Main plot section
st.subheader("Interactive Plot")

# Create and display the plot with click coordinates
plot_img = create_plot_image()

# Display the interactive image
value = streamlit_image_coordinates(
    plot_img,
    key="plot_coords",
    on_click=add_point
)

# Show click information
if value is not None:
    x_pixel, y_pixel = value["x"], value["y"]
    x_data, y_data = pixel_to_data_coords(x_pixel, y_pixel, 800, 600)
    st.markdown(f"""
    <div class="click-info">
        <strong>Last Click:</strong> ({x_data:.2f}, {y_data:.2f})
        <br><small>Pixel coordinates: ({x_pixel}, {y_pixel})</small>
    </div>
    """, unsafe_allow_html=True)

# Method explanations
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="method-box">
        <h3>B-spline Basis Fitting</h3>
        <ul>
            <li><strong>Basis Functions:</strong> Linear combination of smooth basis functions</li>
            <li><strong>Knot Control:</strong> Knot placement directly affects curve shape</li>
            <li><strong>Global Smoothness:</strong> C²-continuous everywhere</li>
            <li><strong>Approximation:</strong> May not pass through data points exactly</li>
            <li><strong>Flexibility:</strong> Degree and smoothing parameters provide control</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="method-box">
        <h3>Akima Spline Interpolation</h3>
        <ul>
            <li><strong>Piecewise Cubic:</strong> Different cubic polynomial per segment</li>
            <li><strong>Exact Interpolation:</strong> Always passes through data points</li>
            <li><strong>Local Method:</strong> Each segment uses only nearby points</li>
            <li><strong>Outlier Robust:</strong> Less sensitive to isolated bad points</li>
            <li><strong>Shape Preserving:</strong> Avoids unwanted oscillations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# Diagnostic plots section
st.markdown("---")
st.header("Diagnostic Plots")

if len(st.session_state.data_points) >= 2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("B-spline Basis Functions")

        # Get B-spline details
        x_data = np.array([p[0] for p in st.session_state.data_points])
        y_data = np.array([p[1] for p in st.session_state.data_points])
        sort_idx = np.argsort(x_data)
        x_data = x_data[sort_idx]
        y_data = y_data[sort_idx]

        try:
            # Fit B-spline and get knots
            if smoothing > 0:
                tck = splrep(x_data, y_data, k=min(degree, len(x_data) - 1), s=smoothing)
            else:
                tck = splrep(x_data, y_data, k=min(degree, len(x_data) - 1))

            t, c, k = tck

            # Plot basis functions
            fig_basis, ax_basis = plt.subplots(figsize=(8, 6))

            x_fine = np.linspace(x_data.min(), x_data.max(), 200)

            # Create B-spline basis functions
            n_basis = len(c)
            colors = plt.cm.Set3(np.linspace(0, 1, min(n_basis, 12)))

            for i in range(n_basis):
                # Create a basis function (set one coefficient to 1, others to 0)
                c_basis = np.zeros_like(c)
                c_basis[i] = 1
                basis_vals = splev(x_fine, (t, c_basis, k))
                color = colors[i % len(colors)]
                ax_basis.plot(x_fine, basis_vals, alpha=0.8, linewidth=2,
                              color=color, label=f'B{i + 1}' if n_basis <= 8 else '')

            # Show knot positions
            internal_knots = t[degree + 1:-degree - 1] if len(t) > 2 * degree + 2 else []
            for knot in internal_knots:
                ax_basis.axvline(x=knot, color='red', linestyle='--', alpha=0.6, linewidth=2)

            ax_basis.set_title(f'B-spline Basis Functions (Degree {degree})')
            ax_basis.set_xlabel('X')
            ax_basis.set_ylabel('Basis Function Value')
            ax_basis.grid(True, alpha=0.3)
            ax_basis.set_facecolor('#fafafa')

            if n_basis <= 8:
                ax_basis.legend(fontsize=10, ncol=2)

            # Add knot information
            knot_text = f'Internal knots: {len(internal_knots)}\nBasis functions: {n_basis}'
            ax_basis.text(0.02, 0.98, knot_text,
                          transform=ax_basis.transAxes, verticalalignment='top',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))

            plt.tight_layout()
            st.pyplot(fig_basis)

        except Exception as e:
            st.error(f"Error creating basis plot: {e}")

    with col2:
        st.subheader("Akima Cubic Segments")

        try:
            # Get Akima interpolation details
            x_data = np.array([p[0] for p in st.session_state.data_points])
            y_data = np.array([p[1] for p in st.session_state.data_points])
            sort_idx = np.argsort(x_data)
            x_data = x_data[sort_idx]
            y_data = y_data[sort_idx]

            if len(x_data) >= 2:
                # Create Akima interpolator
                akima = Akima1DInterpolator(x_data, y_data)

                # Plot individual cubic segments
                fig_segments, ax_segments = plt.subplots(figsize=(8, 6))

                # Plot data points
                ax_segments.scatter(x_data, y_data, s=100, c='black', marker='o',
                                    edgecolors='white', linewidth=2, label='Data Points', zorder=5)

                # Plot each cubic segment
                colors = plt.cm.Set2(np.linspace(0, 1, len(x_data) - 1))

                for i in range(len(x_data) - 1):
                    x_seg = np.linspace(x_data[i], x_data[i + 1], 50)
                    y_seg = akima(x_seg)

                    color = colors[i % len(colors)]
                    ax_segments.plot(x_seg, y_seg, linewidth=3, alpha=0.8,
                                     color=color, label=f'Segment {i + 1}' if len(x_data) <= 9 else '')

                # Mark segment boundaries
                for i in range(1, len(x_data) - 1):
                    ax_segments.axvline(x=x_data[i], color='orange', linestyle=':',
                                        alpha=0.7, linewidth=2)

                ax_segments.set_title('Akima Spline: Individual Cubic Segments')
                ax_segments.set_xlabel('X')
                ax_segments.set_ylabel('Y')
                ax_segments.grid(True, alpha=0.3)
                ax_segments.set_facecolor('#fafafa')

                if len(x_data) <= 9:
                    ax_segments.legend(fontsize=10, ncol=2)

                # Add segment information
                segment_text = f'Cubic segments: {len(x_data) - 1}\nData points: {len(x_data)}'
                ax_segments.text(0.02, 0.98, segment_text,
                                 transform=ax_segments.transAxes, verticalalignment='top',
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.8))

                plt.tight_layout()
                st.pyplot(fig_segments)

        except Exception as e:
            st.error(f"Error creating segments plot: {e}")

    # Additional diagnostic information
    st.markdown("---")
else:
    st.info(
        "Add at least 2 data points to see diagnostic plots showing the underlying mathematical structure of both spline methods!")


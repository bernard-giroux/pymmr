"""pytest configuration for the tests."""
import matplotlib

# no figure windows during the tests (plt.show() does nothing with Agg)
matplotlib.use("Agg")

# scripts that plot results when run directly; they have no tests and run at import
collect_ignore = ["test_cube.py", "test_normal_field.py"]

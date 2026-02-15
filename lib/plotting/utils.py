import io, imageio.v2 as iio
import numpy as np


def fig_to_rgba(fig, dpi=300) -> np.ndarray:
    """
    Render *fig* to an RGBA image in memory and return it as
    a H × W × 4 uint8 NumPy array.  Use `dpi=` if you need higher resolution.
    """
    if isinstance(fig, bytes):
        buf = io.BytesIO(fig)
        img = iio.imread(buf) 
        buf.close()
    else:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        img = iio.imread(buf)                
        buf.close()
    return img[...,:3]
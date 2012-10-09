import numpy as np
from rwt import dwt, idwt
from rwt.utilities import softThreshold, hardThreshold
from rwt.wavelets import daubcqf, waveletlist, waveletCoeffs
import scipy.misc
from enthought.traits.api import HasTraits, Range, on_trait_change, Float, Enum
from enthought.traits.ui.api import View, Item, HGroup, Group, VGroup
from enthought.chaco.api import Plot, ArrayPlotData, gray, jet
from enthought.chaco.tools.api import PanTool, ZoomTool
from enthought.enable.component_editor import ComponentEditor


def vizWavelet(wavelet):
    """Create a visualization of the wavelet"""

    viz = np.log(np.abs(wavelet)+1)
    return (viz/np.max(viz)*255).astype(np.uint8)


class WLapp(HasTraits):

    noise_sigma = Range(0, 50.0, 16.0)
    threshold = Range(0, 100.0, 50.0)
    threshold_type = Enum(('Soft', 'Hard'))

    traits_view = View(
        VGroup(
            HGroup(
                Item('noise_img', editor=ComponentEditor(), show_label=False),
                Item('WL_img', editor=ComponentEditor(), show_label=False),
                Item('denoise_img', editor=ComponentEditor(), show_label=False)
                ),
            HGroup(
                Group(
                    Item('noise_sigma'),
                    Item('threshold_type'),
                    Item('threshold'),
                    Item('wavelet', label='Wavelet Name'),
                    ),
                springy=True
                ),
            ),
        resizable = True,
        title="Wavelets Denoising"
    )

    def __init__(self):

        super(WLapp, self).__init__()

        #
        # Create wavelet traits.
        #
        self.add_trait('wavelet',  Enum(waveletlist(), desc='Name of wavelet'))

        #
        # Create the images
        #
        self._updateNoise()
        WL_img, denoise_img = self._denoise()

        self.plotdata = ArrayPlotData(noise_img=self.noise_image, WL_img=WL_img, denoise_img=denoise_img)

        self.noise_img = Plot(self.plotdata)
        self.noise_img.default_origin='top left'
        self.noise_img.img_plot("noise_img", colormap=gray)
        self.noise_img.title = 'Noised Img'

        self.WL_img = Plot(self.plotdata)
        self.WL_img.default_origin='top left'
        self.WL_img.img_plot("WL_img", colormap=jet)
        self.WL_img.title = 'log of Wavelength Coefficients'

        self.denoise_img = Plot(self.plotdata)
        self.denoise_img.default_origin='top left'
        self.denoise_img.img_plot("denoise_img", colormap=gray)
        self.denoise_img.title = 'Denoised Img'

        for plot in [self.noise_img, self.WL_img, self.denoise_img]:
            plot.tools.append(PanTool(plot))
            plot.tools.append(ZoomTool(plot))

        self.noise_img.range2d = self.denoise_img.range2d

    def _updateNoise(self):
        """Calculate noised image"""

        self.noise_image = scipy.misc.lena().astype(np.float32)
        self.noise_image += np.random.normal(0, self.noise_sigma, size=self.noise_image.shape)

    @on_trait_change('noise_sigma')
    def _updateNoiseImg(self):
        """Update noised image according to new noise sigma and denoise it"""

        self._updateNoise()
        self.plotdata.set_data('noise_img', self.noise_image)
        self._updateDenoise()

    def _denoise(self):
        """Denoise the image using wavelet"""

        c0, c1, r0, r1 = waveletCoeffs(self.wavelet)
        WC, L = dwt(self.noise_image, c0, c1)
        temp = np.sort(WC.ravel())
        threshold = temp[int((len(temp)-1)*(100-self.threshold)/100)]
        if self.threshold_type == 'Hard':
            NWC = hardThreshold(WC, threshold)
        else:
            NWC = softThreshold(WC, threshold)
        return vizWavelet(NWC), idwt(NWC, r0, r1)[0]

    @on_trait_change('wavelet, threshold, threshold_type')
    def _updateDenoise(self):
        """Denoise the image"""

        WL_img, denoise_img = self._denoise()
        self.plotdata.set_data('WL_img', WL_img)
        self.plotdata.set_data('denoise_img', denoise_img)


def main():
    """Main function"""

    app = WLapp()
    app.configure_traits()


if __name__ == '__main__':
    main()
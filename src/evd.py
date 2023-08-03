# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 10:35:49 2022

@author: Ron Simenhois
"""

import numpy as np
from scipy import fftpack
import cv2
import tkinter as tk
from tkinter import filedialog as fd
from tkinter.messagebox import askyesno
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter


MANUAL_MODE = 1

class evd:
    """
    This class handle the Elurian Video Detection acording to the article:
    Using video detection of snow surface movements to estimate weak layer crack propagation speeds
    here: https://urldefense.proofpoint.com/v2/url?u=https-3A__doi.org_10.1017_aog.2023.36&d=DwMFaQ&c=sdnEM9SRGFuMt5z5w3AhsPNahmNicq64TgF1JwNR0cs&r=qyyjeb2qJLbohNPV74KC66ZSWEV219tYajvnHT09Fhw&m=oVZ9Tmu6j9NKa-p5CJqs177iCUGB-F3TEKP6IsA5FqnWAaXyOwba_zODws1ZMKBA&s=5FmXEa6O-fppaULaCnxl-SLdVLwIl-RqOckz7utjodM&e=
    """

    def __init__(self, path=None):

        self.roi = []
        self.x = 0
        self.ix = 0
        self.y = 0
        self.iy = 0
        self.pyr_levels = 3
        self.stab_video = None
        if path is None:
            tk.Tk().withdraw()
            self.src_path = fd.askopenfilename()
        else:
            self.src_path = path
        self.load_video()


    def load_video(self):
        """
        This function load a video file into the memory.

        Returns
        -------
        None. It adds the video as numpy array to the EVD object

        """
        cap = cv2.VideoCapture(self.src_path)
        self.length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.heigth = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.start_frame = 0
        self.end_frame = self.length

        video = []

        for i in tqdm(range(self.length), desc=f'Loading video from: {self.src_path}'):
            ret, frame = cap.read()
            if not ret:
                break
            video.append(frame)
        cap.release()
        self.video = np.array(video)
        tk.Tk().withdraw()
        update_start_end = askyesno('Start End update message',
                                    f'Initiation frame and crack propagation end are set to {self.start_frame} and {self.end_frame}.\n Do you want to update that?')
        if update_start_end:
            self.play_video(self.video,
                            window_name='Click "f" for forward, "b" for backward, "s" and "e" to mark initiation and end propagation frames, "i" and "o" to zoon in and out and Esc to stop')

    def _video_stabilize(self):
        """
        This function uses an optical flow to detect the camera's movements
        between video frames and shift frames to create a "stable camera."
        effect. In this function, the user draws a rectangle around an area
        with texture. The function uses ShiTomasi corner detection to find
        good features to track. These corners are used for the Lucas Kanade
        optical flow algorithm.

        Returns
        -------
        None. It adds the stabilized video in a numpy array as a field to the
              EVD object
        """

        def in_roi(roi, p):

            x, y = p
            return roi['x1'] < x < roi['x2'] and roi['y1'] < y < roi['y2']

        self.get_roi(roi_type='rect')
        roi = {}
        roi['x1'], roi['y1'] = np.array(self.roi).min(axis=0)
        roi['x2'], roi['y2'] = np.array(self.roi).max(axis=0)
        video = self.video[self.start_frame: self.end_frame, ...]
        stab_video = np.zeros_like(video)

        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners=800,
                              qualityLevel=0.01,
                              minDistance=3,
                              blockSize=3)

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15, 15),
                         maxLevel=8,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                   10, 0.03))

        m_dx, m_dy = 0, 0
        # Take first frame and find corners in it
        old_frame = video[0]

        rows, cols, depth = old_frame.shape
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        p0 = cv2.goodFeaturesToTrack(old_gray,
                                     mask=None,
                                     **feature_params)
        p0 = np.expand_dims([p for p in p0.squeeze() if in_roi(roi, p)], 1)

        for idx in tqdm(range(video.shape[0]), desc='Get frames movment'):

            # Get next frame
            frame = video[idx]
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                                   frame_gray, p0,
                                                   None, **lk_params)

            # Select good points
            try:
                good_cur = p1[np.where(st == 1)]
                good_old = p0[np.where(st == 1)]
            except TypeError as e:
                print('TypeError, no good points are avaliabole, error: {0}'.format(e))
                print('Exit video stabilizer at frame {0} out of {1}'.format(idx, self.length))
                break

            dx = []
            dy = []

            # Draw points and calculate
            for i, (cur, old) in enumerate(zip(good_cur, good_old)):
                a, b = cur.ravel()
                c, d = old.ravel()
                dx.append(c - a)
                dy.append(d - b)

            m_dx += np.mean(dx)
            m_dy += np.mean(dy)
            print(m_dx,m_dy)

            M = np.float32([[1, 0, m_dx], [0, 1, m_dy]])

            stab_video[idx] = cv2.warpAffine(frame, M, (cols, rows),
                                           cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP,
                                           cv2.BORDER_CONSTANT).copy()

            marked = stab_video[idx].copy()
            for p in np.squeeze(good_cur):
                marked = cv2.circle(marked, tuple(p.astype(int)), 5, (255,0,0), 2)
            cv2.imshow('stab', marked)
            cv2.waitKey(0)



            # Update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_cur.reshape(-1, 1, 2)
        cv2.destroyAllWindows()

        self.stab_video = np.array(stab_video)


    def get_roi(self, roi_type='poly'):
        """
        This function generates a Region Of Intrest to set the area for
        detection. There are two kinds of ROI it can return: polynomial shared
        ROI or rectangle ROI. Once the ROI is drawn, it saved to the class
        object

        Parameters
        ----------
        roi_type : TYPE: str, optional
            DESCRIPTION. The default is 'poly'.

        Returns
        -------
        None.

        """
        self.roi = []
        if self.stab_video is None:
            video = self.video
        else:
            video = self.stab_video
        if roi_type=='poly':
            self.play_video(video, window_name='Poly ROI',mouse_callback_func=self._get_poly_roi)
        else:
            self.play_video(video, window_name='Rect ROI', mouse_callback_func=self._get_rect_roi)
        self.roi_mask = self._get_poly_mask()


    def _get_poly_mask(self):
        """
        This fuction generate a binary mask from ROI. The mask size is:
        (video's heigth, video width, 1)

        Returns
        -------
        None.

        """

        from matplotlib.path import Path

        x, y = np. meshgrid(range(self.width), range(self.heigth))
        x, y = x.flatten(), y.flatten()
        roi = np.vstack((x, y)).T
        path = Path(self.poly)
        grid = path.contains_points(roi)
        self.mask = grid.reshape(self.heigth, self.width)


    def _get_rect_roi(self, event, x, y, flags, params):
        """
        A mouse callback function that draws a rectangle on a video frame
        and save its corners as ROI for future analasys
        --------------------------------------------------------------
            params: self
                    (int) event: mouse event (left click up, down, mouse move...)
                    (int) x, y: mouse location
                    flags: Specific condition whenever a mouse event occurs.
                            EVENT_FLAG_LBUTTON
                            EVENT_FLAG_RBUTTON
                            EVENT_FLAG_MBUTTON
                            EVENT_FLAG_CTRLKEY
                            EVENT_FLAG_SHIFTKEY
                            EVENT_FLAG_ALTKEY
                    params: user specific parameters (not used)
            return: None
        """
        if event==cv2.EVENT_LBUTTONDOWN:
            self.ix = x
            self.iy = y
            self.drawing = True
        if event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.x = x
                self.y = y
                #self.img = self.current_frame.copy()
                self.current_frame = cv2.rectangle(self.img.copy(),
                                                   (self.ix, self.iy),
                                                   (x, y,),
                                                   (0, 0, 255), 3)
        if event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.roi = ((min(self.ix, self.x), min(self.iy, self.y)),
                        (min(self.ix, self.x), max(self.iy, self.y)),
                        (max(self.ix, self.x), max(self.iy, self.y)),
                        (max(self.ix, self.x), min(self.iy, self.y)),
                        (min(self.ix, self.x), min(self.iy, self.y)))


    def _get_poly_roi(self, event, x, y, flags, params):
        """
        A mouse callback function that draws a rectangle on a video frame
        and save its corners as ROI for future analasys
        --------------------------------------------------------------
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi.append((x, y))
            self.ix = x
            self.iy = y
            self.drawing = True
        if event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_frame = self.img.copy()
                _poly = self.roi + [(x, y), self.roi[0]]
                for p1, p2 in zip(_poly[:-1], _poly[1:]):
                    cv2.line(self.current_frame, p1, p2, (0, 0, 255), 3)
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.roi = self.roi + [(x, y), self.roi[0]]
            self.drawing = False


    def _mark_sample_area(self, event, x,y, flags, params):
        """
        A mouse callback event fuction to get area location for pixel intensity
        location. This fuction save self.sample_rect variabole (size (16,16))
        around the sample point

        Parameters
        ----------
        event : int
            cv2 Mouse events. The inly event that this function respose to is:
            cv2.EVENT_LBUTTONDBLCLK (7)
        x : int
            x axis location on the video frame.
        y : int
            y-axis location on the video frame.
        flags : int
            unused.
        params : dict
            unused.

        Returns
        -------
        None.

        """
        buffer = 8
        if event == cv2.EVENT_LBUTTONDBLCLK:
            x1, x2 = x-buffer, x+buffer
            y1, y2 = y-buffer, y+buffer
            self.sample_rect = ((x1, x2), (y1, y2))


    def dummy_mouse_call_back_func(self, event, x, y, flags, params):
        pass


    def play_video(self,
                   video=None,
                   window_name=None,
                   mode=MANUAL_MODE,
                   mouse_callback_func=None):
        """
        This function plays a set of images on the screen. It is typically used to
        play videos. This function moves one frame forward when the "f" key is
        pressed and one frame back when the "b" key is pressed. It zooms in when
        the "i" key is pressed and out when the "o" key is pressed. The "Esc"
        key ends the video play loop.

        Parameters
        ----------
        video : TYPE - str, optional
            DESCRIPTION. The name of the variabole that holds the video data
            to play. The default is None, the function plays the self.video in this case.
        window_name : TYPE - str, optional
            DESCRIPTION. The default is None.
        mode : TYPE, optional
            DESCRIPTION. The default is MANUAL_MODE - to play by clicking on the
            "f" and "b" keys.
        mouse_callback_func : TYPE - function, optional
            DESCRIPTION. This variable is used to call a specific function that
            handles a mouse event function. For example, it used to draw ROIs.
            The default is None.

        Returns
        -------
        None

        """
        def update_view(img):
            img = cv2.resize(img, None, fx=zoom, fy=zoom,
                             interpolation=cv2.INTER_AREA)
            return img

        if mouse_callback_func==None:
            mouse_callback_func = self.dummy_mouse_call_back_func
        zoom = 1
        if video is None:
            video = self.video
        if window_name is None:
            window_name = 'Video Player'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback_func)
        idx=0
        self.drawing = False
        self.img = video[idx].copy()
        self.current_frame = self.img.copy()
        while True:
            cv2.putText(self.current_frame, str(idx), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5, cv2.LINE_AA)

            cv2.imshow(window_name, self.current_frame)
            k = cv2.waitKey(30) & 0xFF
            if mode==MANUAL_MODE:
                if k==ord('f'):
                    idx = min(idx+1, len(video)-1)
                    self.img = update_view(video[idx])
                    self.current_frame = self.img.copy()
                if k==ord('b'):
                    idx = max(0, idx-1)
                    self.img = update_view(video[idx])
                    self.current_frame = self.img.copy()
                if k==ord('i'):
                    zoom *= 1.33
                    self.img = update_view(video[idx])
                    self.current_frame = self.img.copy()
                if k==ord('o'):
                    zoom *= 0.75
                    self.img = update_view(video[idx])
                    self.current_frame = self.img.copy()
                if k==ord('s'):
                    self.start_frame = idx
                if k==ord('e'):
                    self.end_frame = idx
                if k==27:
                    break

        cv2.destroyAllWindows()
        self.x /=zoom
        self.ix/=zoom
        self.y /=zoom
        self.iy/=zoom

        self.poly = (np.asarray(self.roi)/zoom).astype(int)


    @classmethod
    def down_sample_gausian(cls,
                            data,
                            pyr_levels=3):
        """
        This fuction is a class methoud. It is doing Gausian down sample pyramid

        Parameters
        ----------
        cls : TYPE - evd class
            DESCRIPTION.
        data : TYPE - numpy array or list
            DESCRIPTION - A video array that holds the video.
        pyr_levels : TYPE - int, optional
            DESCRIPTION. The number of pramyd levels. The default is 3.

        Returns
        -------
        The video after it Gausian down sampled.

        """

        video_data = []
        for f in data:
            for i in range(pyr_levels):
                f = cv2.pyrDown(f)
            video_data.append(f)

        return(np.asarray(video_data))

    @classmethod
    def up_sample_gausian(cls, data, pyr_levels=3):
        """
        This fuction is a class methoud. It is doing Gausian up sample pyramid

        Parameters
        ----------
        cls : TYPE - evd class
            DESCRIPTION.
        data : TYPE - numpy array or list
            DESCRIPTION - A video array that holds the video.
        pyr_levels : TYPE - int, optional
            DESCRIPTION. The number of pramyd levels. The default is 3.

        Returns
        -------
        The video after Gausian up sample.

        """

        video_data = []
        for f in data:
            for i in range(pyr_levels):
                f = cv2.pyrUp(f)
            video_data.append(f)

        return(np.asarray(video_data))


    def apply_temporal_filter(self,
                              pyr_levels=3,
                              high_freq = np.nan,
                              low_freq = np.nan,
                              return_size='original'):
        """


        Parameters
        ----------
        pyr_levels : TYPE int, optional
            DESCRIPTION. The downsample pyramid steps to apply on the video
            before applying the temporal  filter. The default is 3.
        high_freq : Type float, optional
            DESCRIPTION. The minimum frequency to filter out from the video
                         signal. The default is NAN; in this case, the minimum
                         frequency is the frequency that alows for two iteratiions
                         from the crack initiation and the appearance of cracks
                         on the snow surface.
        low_freq : Type float, optional
            DESCRIPTION. The minimum frequency to keep after filtering the video
                         signal. The default is NAN; in this case, the minimum
                         frequency is the frequency that alows for one iteratiions
                         from the crack initiation and the appearance of cracks
                         on the snow surface.
        return_size : TYPE str, optional
            DESCRIPTION. 'original' to return the filtered signal as the
                         original video size, everything else will return
                         in the downsampled size. The default is 'original'.

        Returns
        -------
        filtered_data : TYPE numpy array
            DESCRIPTION. a numpy array of the video after filtering

        """
        self.pyr_levels = pyr_levels
        frequancies = fftpack.fftfreq(self.end_frame - self.start_frame,
                                      d=1.0/self.fps)
        video = self.video#[self.start_frame: self.end_frame]
        gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in video]
        gray = [np.where(self.mask, f, 0).astype(np.uint8) for f in gray]
        video_data = self.down_sample_gausian(gray, pyr_levels)
        fft_signal = fftpack.fft(video_data, axis=0)
        # get the frequancies to filter out:
        high_freq = 2 * (self.end_frame - self.start_frame) / self.fps
        low_freq = (self.end_frame - self.start_frame) / self.fps

        high_bound = np.abs(frequancies-high_freq).argmin()
        low_bound = max(np.abs(frequancies-low_freq).argmin(), 1)
        # Zero the frequancies we dont need
        fft_signal[:low_bound] = 0
        fft_signal[high_bound:-high_bound] = 0
        fft_signal[-low_bound:] = 0
        # Reverse Fourier transform after zeroing out unwanted frequancies
        filtered_data = np.real(fftpack.ifft(fft_signal,axis=0))
        if return_size=='original':
            filtered_data = self.up_sample_gausian(filtered_data, pyr_levels)

        return filtered_data


    @classmethod
    def filter_1D_signal(cls, signal, fps=50, high_f=0):

        # The FFT of the signal
        sig_fft = fftpack.fft(signal)
        # And the power (sig_fft is of complex dtype)
        power = np.abs(sig_fft)**2
        # The corresponding frequencies
        sample_freq = fftpack.fftfreq(len(signal), d=1.0/fps)
        # Find the peak frequency: we can focus on only the positive frequencies
        pos_mask = np.where(sample_freq > 0)
        freqs = sample_freq[pos_mask]
        # Get low frequancy
        low_freq = freqs[power[pos_mask].argmax()]
        high_freq = freqs[round(2*len(signal)/fps)]
        low_bound = np.abs(sample_freq-low_freq).argmin()
        high_bound = np.abs(sample_freq-high_freq).argmin()
        # Get High frequency
        #high_freq = freqs[int(power[pos_mask].argmax()+2*len(signal)/fps)]

        fft_signal_f = sig_fft.copy()
        fft_signal_f[:low_bound] = 0
        fft_signal_f[high_bound:-high_bound] = 0
        fft_signal_f[-low_bound:] = 0


        filtered_sig = fftpack.ifft(fft_signal_f)

        return filtered_sig.real


    @classmethod
    def save_video(cls, video=None, fps=30):
        """
        This fuction save a 5D tensor as video file

        Parameters
        ----------
        video : TYPE list or numpy array, optional
            DESCRIPTION. the video tensor. The default is None.
        fps : TYPE int, optional
            DESCRIPTION. the fps of the video to sace asThe default is 30.

        Returns
        -------
        None.

        """

        if video is None:
            return
        video = np.array(video)
        length, height, width, ch = video.shape

        tk.Tk().withdraw()
        save_file = fd.asksaveasfilename()
        if save_file==None:
            return
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename=save_file, fourcc=fourcc, fps=fps,
                              frameSize=(width, height),isColor=True)
        for frame in video:
            out.write(frame)
        out.release()
        print('Video file is saved')
        return save_file


    def detect_changes(self,
                       data,
                       threshold=3,
                       **kwargs):
        """
        This function detects extreme pixel intensity changes by pixel.
        It maps the video frame where the second derivative of the pixel
        intensity is 0, and the absolute value of the first derivative of
        the pixel intensity is > the first derivative mean + 3 times the
        standard deviation.

        Parameters
        ----------
        video_data : np.array of float
            The videp in gray scale after temporal filtering.
        threshold : float, optional
            The number of standard deviations to set the detection filter.
            The default is 3.
        kwargs: options dict
            possibole keys:
                'gausian kernel size': kernel size for changs development over
                                       time display. Default is 11
                'methoud maxima': defime threshold for detection as number of
                                  stanf=datd deviations from the mean or 0.97
                                  from the maximum change over time
                'std to outlier': The number of standard deviation from the
                                  mean to use as outlier for threshold for
                                  detected changes. The deafult is 3
        Returns
        -------
        detected : TYPE
            DESCRIPTION.

        """
        gausian_kernel = kwargs.get('gausian kernel size', 11)
        maxima_methoud = kwargs.get('methoud maxima', 'from max')
        diff_data = np.diff(data, axis=0)
        # Get points where diff_data reach minima or maxima:
        diff_2_data = np.diff(diff_data, axis=0)
        # Find where the second derivative crosses the zero
        diff_maxima = np.diff(np.signbit(diff_2_data), axis=0).astype(int)
        # Add buffer to keep it on the same length as the video data
        diff_maxima = np.concatenate([diff_maxima[:1,...],
                                      diff_maxima,
                                      diff_maxima[-1:,...]],
                                     axis=0)
        # Find intensity change maxima
        diff_maxima = np.bitwise_and(self.mask, diff_maxima)
        diff_maxima[:self.start_frame: self.end_frame] = 0
        # find outlier pixel intensity changes
        diff_data_maximas = np.abs(diff_data * diff_maxima)
        if maxima_methoud == 'from max':
            threshold = diff_data_maximas.max(axis=0) * 0.97
        else: #from std
            std_to_outlier = kwargs.get('std to outlier', 3)
            _mean = diff_data[self.start_frame: self.end_frame].mean(axis=0)
            _std = diff_data[self.start_frame: self.end_frame].std(axis=0)
            threshold = _mean + _std * std_to_outlier
        detected = np.where(diff_data_maximas > threshold, 1, 0)
        _detected = np.maximum.accumulate(detected[self.start_frame:self.end_frame])

        _detected = np.array([gaussian_filter(d, gausian_kernel) for d in _detected.astype(float)])
        detected = np.zeros_like(detected, dtype=float)
        end_frame = self.start_frame + len(_detected)
        detected[self.start_frame: end_frame] =_detected.astype(float)
        detected[end_frame:] = \
            np.where(diff_data[end_frame:] > threshold, 1, 0).astype(float)

        return detected

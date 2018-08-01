#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Eudie and Anuj

"""
In this class I am trying in structure dynamic traffic signal module. Which will be used in product.
We are trying to optimize the signal using simulation. We are using OpenStreetMap to get map and SUMO for optimization.

"""


from __future__ import absolute_import
from __future__ import print_function
import one_camera_analysis


class VideoAnalysis:
    """
    Here we are going to combine all analysis and expose to product
    """

    def __init__(self, name):
        self.name = name

    def reader(self, dict_of_rtsp):
        """
        Here we read all rtsp dict of n cameras and save m recent frames in n*m array of images.
        We will also make the folder for each rtsp to save the results
        :param dict_of_rtsp: {'name_of_camera': 'rtsp link'}
        """

        # TODO: Build the reader that read all rtsp and save in shared memory

    def object_detector(self):
        """
        It takes the image saved from reader of all camera and do the detection and saved it in shared memory again
        :return:
        """

        # TODO: Find the fastest detector

    def parallel_analysis(self):
        """
        Get result of all camera in parallel
        """

        # For each camera run one_camera_analysis and save in respective folder
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Eudie and Anuj

"""
In this class I am trying in structure dynamic traffic signal module. Which will be used in product.
We are trying to optimize the signal using simulation. We are using OpenStreetMap to get map and SUMO for optimization.

"""


from __future__ import absolute_import
from __future__ import print_function


class OneCameraAnanlysis:
    """
    Here we will parallalize all analysis for one camera
    """

    def __init__(self, object_dict, frame, output_folder):
        self.object_dict = object_dict
        self.frame = frame
        self.output_folder = output_folder

    def analysis_1(self):
        """
        Analysis
        :return: bounding box
        """
        # TODO: incude all analysis

    def analysis_2(self):
        """
        Analysis
        :return: bounding box
        """
        # TODO: incude all analysis

    def combine_all_analysis(self):
        """
        Do all analysis in parallel
        :return: bbox of all
        """
        # TODO: parallelize all analysis

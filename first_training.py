# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 11:10:39 2019

@author: Anchu
"""

from imageai.Prediction.Custom import ModelTraining

model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory("idenprof")
model_trainer.trainModel(num_objects=5, num_experiments=200, enhance_data=True, batch_size=32, show_network_summary=True)

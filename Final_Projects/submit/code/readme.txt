Explain:
	main.py: run to run the both English and Chinese experiment —— by Jiaqi Han 
	denoise.py: denoise for English and Chinese —— by Lihua An 
	evaluation.py: ground truth and evaluation code for English and Chinese —— by Jiaqi Han
	make_template.py: used to manually make templates —— by Jiaqi Han
	our_implementation.py: our implementation of cv2 functions —— by Lihua An
	recognition.py: recognition algorithms, the first piece of commented out code is the comparasion of different matching algorithms —— by Jiaqi Han
	segment.py: lines & words segmentation algorithms, note that some code is borrowed from other commercial OCR projects done by Zui Chen and Jiaqi Han —— by Zui Chen
	shear.py: shear, skew detection algorithms for English —— by Lihua An
	utils.py: other miscellaneous methods —— by Jiaqi Han

Note that our template matching implementation is rather slow, but correct. If you want to increase the speed, modify line 339 in recognize.py to use cv2 built in function.

Other problems please contact Jiaqi Han: hanjq2022@shanghaitech.edu.cn
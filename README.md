<html>
<body>
<h1>Detector2</h1>
<font size=3><b>
This is a simple python class Detector2 based on DefaultDetector class of detectron2.<br>
</b></font>
<br>
<h2>1 Installation </h2>
<h3>
1.1 detectron2
</h3>
<font size=2>
 We have downloaded <a href="https://github.com/facebookresearch/detectron2">detectron2</a>.
and installed pytorch-cpu and tochvision-cpu in the following way.<br>

<br>
<table style="border: 1px solid red;">
<tr><td>
<font size=2>
git clone https://github.com/facebookresearch/detectron2.git<br>
pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install Cython<br>
pip install git+https://github.com/facebookresearch/fvcore<br>
pip install opencv-python==4.3.0<br>
pip install PyYAML==5.3.1<br>
pip install requests==2.24.0<br>
pip install requests-oauthlib==1.3.0<br>
</font>
</td></tr>
</table>

<h3>
1.2 cocoapi
</h3>
Please change directory to the detectron2 folder, and clone cocoapi.git in the following way.
<pre>
\detectron2>git clone https://github.com/cocodataset/cocoapi.git
</pre>

Modify cocoeval.cpp file.<br>
<pre>
cocoeval.cpp  
//2020/08/5 localtime_r(&rawtime, &local_time); No localtime_r in vs2019
  localtime_s(&rawtime, &local_time);
</pre>

1.3 Modify setup.py
</h3>
Please modify extra_compile_args in detectron2\cocoapi\PythonAPI\setup.py file in the following way. 
<pre>
ext_modules = [
    Extension(
        'pycocotools._mask',
        sources=['../common/maskApi.c', 'pycocotools/_mask.pyx'],
        include_dirs = [np.get_include(), '../common'],
        #2020/07/05 extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
        extra_compile_args=[],
    )
]
</pre>

<h3>
1.4 Run the following command.
</h3>

detectron2\cocoapi\PythonAPI>python setup.py build_ext install

<br>
Please clone <a href="https://github.com/atlan-antillia/Detector2.git">Detector2.git</a> in a 
folder under the detectron2 folder made by git cloning detetron2.
<pre>
>git clone  https://github.com/atlan-antillia/Detector2.git
</pre>


<h2>2 Detector2 class</h2>
 We have defined Detector2 class and added some new classes in this tiny-project.<br>
<br>
<a href="./Detector2.py">Detector2</a> <br>

<a href="./DetectronModelZoo.py">DetectronModelZoo</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="./CocoInstanceSegmentation.py">CocoInstanceSegmentation</a> <br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="./CocoPanopticSegmentation.py">CocoPanopticSegmentation</a> <br>
<a href="./FilteredVisualizer.py">FilteredVisualizer</a><br>
<a href="./FiltersParser.py">FiltersParser</a><br>

<br>


<h2>3 Sample programs</h2>
<h3>
3.1 InstanceSegmentation.py
</h3>
Please run the following command

<pre>
>python Detector2\InstanceSegmentation.py Detector2\images\img.png Detector2\detected [car]
</pre>
<img src="./detected/instance_seg_img.png" width="80%" height="auto">

<h3>
3.2 PanopticSegmentation.py
</h3>
<pre>
>python Detector2\PanoticSegmentation.py Detector2\images\img.png Detector2\detected 
</pre>

<img src="./detected/panoptic_img.png" width="80%" height="auto">
<br>
<br>
<br>
<h2>
Citing Detectron2
</h2>
If you use Detectron2 in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry.
<pre>
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
</pre>
</body>
</html>


# openvino_binary
A quick demo on how to build a binary executable with Python and OpenVINO


#Usage

```
python -m venv venv
venv/Script/activate
```
```
python -m pip install --upgrade pip
pip install -r requirements.txt
```
```
###Test and see results
python main.py 
```

###Build binary 
```
pyinstaller main.py --collect-all openvino --add-data _internal/model:model --add-data _internal/img/intel_rnb.jpg:img/
```

Note: The executable and package is in the dist folder.

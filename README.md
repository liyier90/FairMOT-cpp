# FairMOT-cpp

```
git clone https://github.com/liyier90/FairMOT-cpp.git && cd FairMOT-cpp
git submodule update --init --recursive
```

Download [fairmot_dla34.pth](https://drive.google.com/file/d/1iqRQjsG9BawIl8SlFomMg5iwkb6nqSpi/view) to the `weights/` folder.

```
cd python
pip install -r requirements.txt
python convert_to_jit.py
```

_Older versions of the required Python packages may work as well._

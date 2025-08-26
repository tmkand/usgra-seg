# usgra-seg
Ultraound-Guided Regional Anaesthesia semantic SEGmentation machine learning models

This project served to create and train different neural network architectures for the semantic segmentation of nerves and blood vessels in ultrasound images used during ultrasound-guided regional anaesthesia (USGRA). 

## Dataset

The [dataset](dataset/) consists of 244 image-mask-pairs in PNG format. 40 image-mask-pairs have been isolated into the [test](dataset/test/) subfolder for testing the models efficiency after training. They have been acquired from volunteers in early 2025 in a German Anaesthesiology Department by a single examiner (TMK) using a Mindray TE7 ultrasound machine with a L14-6Ns linear probe. The examined regions were (1) supraclavicular to interscalene cervical region, (2) infraclavicular to axillary plexus region, (3) elbow region, (4) femoral nerve region in the groin, (5) popliteal sciatic nerve region. 

Masks have been created by the examiner (TMK) using the [LabelMe](https://github.com/wkentaro/labelme) annotation tool.

## License
- All **source code** in this repository is licensed under the [MIT License](LICENSE).
- The **dataset** located in the `dataset/` directory is licensed under [CC-BY 4.0](dataset/LICENSE)

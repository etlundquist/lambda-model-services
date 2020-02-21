#!/bin/bash

# specify the project root and dependency sub-folders
export BASE_PATH="/Users/ericlundquist/Repos/lambda-model-services/lambda"
export DEPS_PATH="python"

# install all dependencies using a AWS Linux docker run-time
rm -rf ${BASE_PATH}/${DEPS_PATH} && mkdir -p ${BASE_PATH}/${DEPS_PATH}
docker run --rm -v ${BASE_PATH}:/lambda -w /lambda lambci/lambda:build-python3.6 pip install -r requirements.txt -t /lambda/${DEPS_PATH}

# build the zip file and upload the layer to AWS
zip -r ${BASE_PATH}/dependencies.zip ${BASE_PATH}/${DEPS_PATH}

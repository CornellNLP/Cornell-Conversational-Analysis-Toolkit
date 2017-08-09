make html
rsync -a build/html/ ~/dev/qema.github.io/convokit/
pushd .
cd ~/dev/qema.github.io
nanosite p
popd

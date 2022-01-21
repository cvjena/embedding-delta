docker run -v "$(pwd)"/data/test:/data/test -v "$(pwd)"/predictions:/predictions -it stss-eval


# docker run -v "$(pwd)"/data/test:/data/test -v "$(pwd)"/predictions:/predictions -it --entrypoint /bin/bash stss-eval

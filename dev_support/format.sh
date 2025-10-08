# if argument contains "check", only check the code style
check_flag=""
if [ "$1" == "check" ]; then
    check_flag="--check"
fi


lint () {
    files_to_lint=$(git diff $1 --name-only | grep ".py")
    echo $files_to_lint
    if [ "x" = "x$files_to_lint" ]; then
        echo "No files to lint"
        return
    fi

    autoflake $check_flag --in-place --recursive --remove-all-unused-imports --remove-unused-variables $files_to_lint
    isort --profile black $check_flag $files_to_lint
    black $check_flag $files_to_lint
    blackdoc $files_to_lint
}

if [ "x" != "x$2" ]; then
    lint $2
else
    lint $(git merge-base origin/master HEAD)
fi
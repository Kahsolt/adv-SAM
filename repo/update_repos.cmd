@ECHO OFF

FOR /D %%p IN (*) DO (
  ECHO git pull %%p
  PUSHD %%p
  git stash
  git pull --rebase
  git stash pop
  POPD
)

ECHO Done!
ECHO.

PAUSE

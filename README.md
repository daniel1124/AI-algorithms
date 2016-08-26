# Setup
Clone this repository recursively:
`git clone --recursive https://github.gatech.edu/omscs66601/assignment_1.git`

(If your version of git does not support recurse clone, then clone without the option and run `git submodule init` and `git submodule update`).

# Keeping your code upto date
After the clone, we recommend creating a branch and developing your agents on that branch:

`git checkout -b develop`

(assuming develop is the name of your branch)

Should the TAs need to push out an update to the assignment, commit (or stash if you are more comfortable with git) the changes that are unsaved in your repository:

`git commit -am "<some funny message>"`

Then update the master branch from remote:

`git pull origin master`

This updates your local copy of the master branch. Now try to merge the master branch into your development branch:

`git merge master`

(assuming that you are on your development branch)

There are likely to be merge conflicts during this step. If so, first check what files are in conflict:

`git status`

The files in conflict are the ones that are "Not staged for commit". Open these files using your favourite editor and look for lines containing `<<<<` and `>>>>`. Resolve conflicts as seems best (ask a TA if you are confused!) and then save the file. Once you have resolved all conflicts, stage the files that were in conflict:

`git add -A .`

Finally, commit the new updates to your branch and continue developing:

`git commit -am "<funny message vilifying TAs for the update>"`

# Play against a test agent
To play against a test agent, use `python submit.py play_isolation`. (If you are a TA, add the option `--provider udacity`).

# Submit your code
To submit your code to have it evaluated for a grade, use `python submit.py assignment_1`.  You may submit as many times as you like.  The last submission before the deadline will be used to determine your grade.

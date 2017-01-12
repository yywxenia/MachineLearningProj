README: Description of python scripts and support files for project 4
Yiwei Yan

==================================================================
PART 1: SOURCES OF DATA

The datasets we used are created by functions at MDP_QL_PLOT.py in the folder named “P4_Code”.

1. Case 1: Deterministic data [function graduate_path0()] and Non-Deterministic data [function graduate_path1()] with 4 states and 2 actions;

2. Case 2: Non-Deterministic data [fix_machine(S, p1, p2, p3, p4, p5, r1, r2, r3, r4, r5)] with any number of states (we used 4 to 100 in the report) and 5 actions.


————————————————————————————————————
PART 2: MAIN FUNCTION MODULES

All python scripts are saved in folder named ”P4_Code". There are totally 3 .py scripts in this folder.

0. The code depends on the MDP toolbox. I did not use the original version downloaded on the internet. Instead, I modified the toolbox to support more functionalities. The modified toolbox is in “Modified_mdptoolbox” subdirectory. Please add this to $PYTHONPATH by:
   $export PYTHONPATH=/path/to/Modified_mdptoolbox:$PYTHONPATH

1. MDP_QL_PLOT.py: this script includes all the data-creating functions, algorithms, and plottings for the report 4.

2. Case1_Graduated.py: run for getting all results at case 1.

3. Case2_FixMachine.py: run for getting all results at case 1.

Please check more parameters details in those python files.


————————————————————————————————————
PART 3: REFERENCES: GRAPHS AND RESULT FILES

Please find those files in folder named ”P4_Supporting Files".

1. Sub-folder named “Case1_Results” contains all results .txt files and support graphs for MDP case 1.

2. Sub-folder named “Case2_Results” contains all results .txt files and support graphs for MDP case 2.

Thank you!



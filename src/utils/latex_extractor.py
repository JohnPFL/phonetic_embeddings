

import pandas as pd 
from os.path import join
import os 
import argparse 
import subprocess
from models.classifier import bold_max
with open('src/src.txt', "r") as src: 
    src = src.read()

def main():
    # The default hyperparameters were chosen based on a hyperparameters analysis 
    parser = argparse.ArgumentParser(description='Latex generation')
    parser.add_argument('--results_path', default='/home/sperduti/vgm/Datasets/results/_datasetname_hs_batch_size_32_hardness_1000_lr_1e-05_visual_lr_1e-05_kernel_heights_[5, 6, 7]_visual_kernel_heights_[5, 6, 7]_seed_43', type=str, help='name of the dataset to be selected')
    args = parser.parse_args()

    results_path = args.results_path
    results_path_n = args.results_path.split('/')[-1]

    # These files contain the experimental results for the clean, adversarial, and augmented frameworks respectively.
    results_clean = pd.read_pickle(results_path + '_cl.pkl')
    results_adv = pd.read_pickle(results_path + '_adv.pkl')
    results_aug = pd.read_pickle(results_path + '_aug.pkl')

    # Final Latex path for the expriments
    latex_path = join('Datasets/results/', 'latex_tables', results_path_n)
    os.makedirs(latex_path, exist_ok=True)

    conditions = ['clean', 'adv', 'aug']
    for condition in conditions:
        # Loading each one of the 3 variables containing results
        results = eval(f'results_{condition}')
        for column in results.columns:
            # Formatting function to highlight the maximum values in each column 
            results[column] = bold_max(results[column])
        # Exporting bolded results in latex into the right path (with the name of the experiment -results_path_n-)
        lat_results = results.to_latex(escape=False)
        with open(join(src, latex_path, condition + ".tex"), 'w') as lr:
            lr.write(lat_results)

    # Open the file in read mode
    with open(join('Datasets/results/', 'latex_tables', 'main.tex'), 'r') as f:
        # Read the contents of the file
        contents = f.read()

    # New section for our latex main. Hyperparameters as title
    section_name = "{" + results_path_n.replace('_', ' ') + "}"
    # Loading and preparing the 3 latex tables to be imported in main
    clean = "{" + latex_path + "/clean.tex}"
    adv =  "{" + latex_path + "/adv.tex}"
    aug = "{" + latex_path + "/aug.tex}"
    tables = f"\input{clean}\\ \input{adv} \\ \input{aug}"

    # Inserting new conteng at the end of the document for every new iteration
    new_contents = contents.replace("\\end{document}", f'\\section{section_name}' +"\n\n" + tables + "\n\\end{document}")

    # Write the modified contents to the file
    with open(join('Datasets/results/', 'latex_tables', 'main.tex'), 'w') as f:
        f.write(new_contents)

    # Running pdflatex to generate the final main pdf 
    #subprocess.call(['pdflatex', '-output-directory', 'Datasets/results/pdf', '-halt-on-error', 'Datasets/results/latex_tables/main.tex'])

if __name__ == '__main__':
    main()
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from config import YEAR, BASE_URL, DATA_FILEPATH, PLOTS_FILEPATH

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, roc_auc_score


# YEAR variable
YEAR = YEAR

# FILE locations 
data_filepath = DATA_FILEPATH
plots_filepath = PLOTS_FILEPATH

def data_request():
    base_url = BASE_URL

    if os.path.exists(data_filepath):
        hockey_shots = pd.read_csv(data_filepath)


    
    else:
        hockey_shots = pd.read_csv(base_url)
        hockey_shots.to_csv(data_filepath)


    return hockey_shots


def build_rink_scatter():
    plt.xlim(0,100)
    plt.ylim(-42.5,42.5)
    
    # PLOT CENTER ICE LINE FOR REF
    plt.axvline(x=0, color='red', linestyle='-', linewidth=2)
    plt.axvline(x=89, color='red', linestyle='-', linewidth=2)
    plt.axvline(x=-89, color='red', linestyle='-', linewidth=2)
    
    plt.axvline(x=25, color='blue', linestyle='--', linewidth=2)
    plt.axvline(x=-25, color='blue', linestyle='--', linewidth=2)
    
    plt.scatter(69,22, color='blue', s=25)
    plt.scatter(69, -22, color='blue', s=25)


def generate_heatmap(df,name,p):
    heat_filename = f'plots/{name}_heatmap_{YEAR}.png'

    """ if not os.path.exists(heat_filename) and not p: # FILE doesn't exist and NOT a prediction
        heatmap_pivot = df.pivot(index="xCordAdjusted", columns="yCordAdjusted", values="")
        # FIX
        plt.imshow(df, cmap="plasma", interpolation='nearest')
    
        plt.colorbar(label='Value')
        plt.title(f"{name} {YEAR} Heatmap")
    
        plt.savefig(heat_filename, format='png')
        plt.close() """

    if p: #GENERATE predictive heatmap
        heatmap_pivot = df.pivot(index="xCordAdjusted", columns="yCordAdjusted", values="probs")
        
        sns.heatmap(
            heatmap_pivot,
            cmap="plasma",
            vmin=0, vmax=1,
            cbar_kws={'label': 'Predicted Goal Probabilites'}
        )

        plt.title("Predicted Hockey Goal Probabilites")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")

        plt.savefig(heat_filename, format='png')
        plt.close()


def generate_shot_map(dataset, name, p):
    shot_map_filename = f'plots/{name}_shot_plot_{YEAR}.png'

    if not os.path.exists(plots_filepath) and not p:  # FILE doesn't exist and NOT a prediction
        build_rink_scatter()

        #SEPERATE goals/no goals
        no_goals = dataset.query("goal == 0")
        goals = dataset.query("goal ==  1")
        
        #SCATTER pts
        plt.scatter(no_goals["xCordAdjusted"], no_goals["yCordAdjusted"], s=5, c="red")
        plt.scatter(goals["xCordAdjusted"], goals["yCordAdjusted"], s=5, c="green")
        
        # SAVE fig
        plt.savefig(shot_map_filename, format='png')
        plt.close()

    
    if p:  #GENERATE a predictive shot map
        build_rink_scatter()

         #SEPERATE goals/no goals
        no_goals = dataset.query("goal_predict == 0")
        goals = dataset.query("goal_predict ==  1")
        
        #SCATTER pts
        plt.scatter(no_goals["xCordAdjusted"], no_goals["yCordAdjusted"], s=5, c="red")
        plt.scatter(goals["xCordAdjusted"], goals["yCordAdjusted"], s=5, c="green")
        
        # SAVE fig
        plt.savefig(shot_map_filename, format='png')
        plt.close()
        





# -------------------------------------------------------------------------------------------- #

# Machine Learning Functs 

def linear_regression(hockey_data):
    print("GENERATING REGRESSION")

    x = hockey_data[["xCordAdjusted", "yCordAdjusted"]]
    y = hockey_data["goal"]
    
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

    model = LogisticRegression(max_iter=10000, class_weight="balanced")
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:,1]

    # VIEW scores
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("AUC:",roc_auc_score(y_test, y_prob))
    print("Model Coeffs:", model.coef_)

    return model



def generate_machine_learning_heatmap(model):
    coords_list = []
    for x in range(0,90):
        for y in range(-42,43):
            coords_list.append([x,y])

    hockey = pd.DataFrame(coords_list, columns=["xCordAdjusted","yCordAdjusted"])

    probabilites = model.predict_proba(hockey)
    goal_predictions = model.predict(hockey)

    hockey["probs"] = probabilites[:,1]
    hockey["goal_predict"] = goal_predictions


    generate_heatmap(hockey, "predict", True)
    generate_shot_map(hockey, "predict", True)



# main function

def main():
    YEAR = 2024

    hockey_shot_data = data_request()

    model = linear_regression(hockey_shot_data)

    generate_machine_learning_heatmap(model)

if __name__ == "__main__":
    main()
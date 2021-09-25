from django import forms

weights_choices = [('uniform', 'uniform'),
                   ('distance', 'distance')]
algorithm_choices = [('auto', 'auto'),
                     ('ball_tree', 'ball_tree'),
                     ('kd_tree', 'kd_tree'),
                     ('brute', 'brute')]


class InputForm(forms.Form):
    n_neighbors = forms.IntegerField(help_text="Enter Number of Neighbors")
    leaf_size = forms.IntegerField(help_text="Enter Leaf Size")
    weights = forms.CharField(label='Select Weights Type', widget=forms.Select(choices=weights_choices))
    algorithm = forms.CharField(label='Select Algorithm Type', widget=forms.Select(choices=algorithm_choices))

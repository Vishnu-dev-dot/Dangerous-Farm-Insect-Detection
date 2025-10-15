from django import forms

class InsectImageForm(forms.Form):
    image = forms.ImageField(
        label="Upload insect image",
        widget=forms.ClearableFileInput(attrs={
            'class': 'form-control',
            'accept': 'image/*'
        })
    )

class PesticideCalculatorForm(forms.Form):
    area_sqft = forms.FloatField(
        label="Area (sq.ft)",
        min_value=0.1,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter area in square feet'
        })
    )

    insect_class = forms.ChoiceField(
        label="Insect / Pest",
        choices=[],  # views will set choices dynamically from CLASS_NAMES
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

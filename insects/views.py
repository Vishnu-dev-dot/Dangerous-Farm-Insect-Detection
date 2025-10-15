from django.shortcuts import render
from django.conf import settings
from .forms import InsectImageForm, PesticideCalculatorForm
from .utils import predict_from_path, calculate_pesticide_for_area, CLASS_NAMES, PESTICIDE_MAP
from django.core.files.storage import FileSystemStorage
from PIL import Image
import os

# def home(request):
#     return render(request, 'insects/home.html', {'class_count': len(CLASS_NAMES)})
from .utils import CLASS_NAMES  # already imported above

def home(request):
    return render(
        request,
        'insects/home.html',
        {
            'class_count': len(CLASS_NAMES),
            'classes': CLASS_NAMES,   # <-- add this
        }
    )

def gallery(request):
    return render(request,'insects/gallery.html')

def upload_and_predict(request):
    result = None
    uploaded_url = None
    form = InsectImageForm(request.POST or None, request.FILES or None)
    if request.method == 'POST' and form.is_valid():
        img = form.cleaned_data['image']
        fs = FileSystemStorage()
        filename = fs.save(img.name, img)
        filepath = fs.path(filename)
        uploaded_url = fs.url(filename)  # <-- for preview

        try:
            # Force RGB
            pil_img = Image.open(filepath).convert('RGB')
            pil_img.save(filepath)

            # Predict using utils
            result = predict_from_path(filepath)
        except Exception as e:
            result = {'error': str(e)}

    return render(request, 'insects/upload.html', {
        'form': form,
        'result': result,
        'uploaded_url': uploaded_url,  # <-- pass to template
    })


def calculate_dosage(request):
    output = None
    all_outputs = None  # <-- for "all classes"
    choices = [(c, c) for c in CLASS_NAMES]
    if request.method == 'POST':
        form = PesticideCalculatorForm(request.POST)
        form.fields['insect_class'].choices = choices
        if form.is_valid():
            area = form.cleaned_data['area_sqft']
            insect = form.cleaned_data['insect_class'] or CLASS_NAMES[0]
            output = calculate_pesticide_for_area(area, insect)

            # Build table for all classes
            all_rows = []
            for label in CLASS_NAMES:
                row = calculate_pesticide_for_area(area, label)
                row['recommendation'] = PESTICIDE_MAP.get(label, 'No recommendation available — update PESTICIDE_MAP')
                all_rows.append(row)
            all_outputs = all_rows
    else:
        form = PesticideCalculatorForm()
        form.fields['insect_class'].choices = choices

    return render(request, 'insects/calculator.html', {
        'form': form,
        'output': output,
        'all_outputs': all_outputs,  # <-- pass to template
    })

# def upload_and_predict(request):
#     result = None
#     form = InsectImageForm(request.POST or None, request.FILES or None)
#     if request.method == 'POST' and form.is_valid():
#         img = form.cleaned_data['image']
#         fs = FileSystemStorage()
#         filename = fs.save(img.name, img)
#         filepath = fs.path(filename)

#         try:
#             # Force RGB
#             pil_img = Image.open(filepath).convert('RGB')
#             pil_img.save(filepath)

#             # Predict using utils
#             result = predict_from_path(filepath)
#         except Exception as e:
#             result = {'error': str(e)}

#     return render(request, 'insects/upload.html', {'form': form, 'result': result})

# def calculate_dosage(request):
#     output = None
#     choices = [(c, c) for c in CLASS_NAMES]
#     if request.method == 'POST':
#         form = PesticideCalculatorForm(request.POST)
#         form.fields['insect_class'].choices = choices
#         if form.is_valid():
#             area = form.cleaned_data['area_sqft']
#             insect = form.cleaned_data['insect_class'] or CLASS_NAMES[0]
#             output = calculate_pesticide_for_area(area, insect)
#     else:
#         form = PesticideCalculatorForm()
#         form.fields['insect_class'].choices = choices
#     return render(request, 'insects/calculator.html', {'form': form, 'output': output})

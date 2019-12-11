import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

All_RMS_Errors = {
    'MIT-LRN': {
        # Original MIT paper model, LRN instead of CrossMapLRN2d
        # Batch 128 on Alienware 51m NVIDIA 2080Ti
        'RMS_Errors': [2.3286047396759257, 2.1853567796821314, 2.1260477360736307, 2.1231018788556404,
                       2.165244323131456, 2.1346357934652915, 2.0952607714737033, 2.1207367246393076,
                       2.1513301759577472, 2.145653521204925, 2.1244738083681005, 2.157344966617782, 2.200293857488523,
                       2.1848297350023445, 2.1600034351066686, 2.196173098076087, 2.212923449354037, 2.2122061388558323,
                       2.216721373463767, 2.228971885432312, 2.208697491583603, 2.2454612007218073, 2.2529483607894396,
                       2.2494198072173086, 2.2300056850340995, 2.2392063010035543, 2.2757812097237746,
                       2.2527829522964695, 2.262061535566544, 2.2626777626799512],
        'Best_RMS_Errors': [2.3286047396759257, 2.1853567796821314, 2.1260477360736307, 2.1231018788556404,
                            2.1231018788556404, 2.1231018788556404, 2.0952607714737033, 2.0952607714737033,
                            2.0952607714737033, 2.0952607714737033, 2.0952607714737033, 2.0952607714737033,
                            2.0952607714737033, 2.0952607714737033, 2.0952607714737033, 2.0952607714737033,
                            2.0952607714737033, 2.0952607714737033, 2.0952607714737033, 2.0952607714737033,
                            2.0952607714737033, 2.0952607714737033, 2.0952607714737033, 2.0952607714737033,
                            2.0952607714737033, 2.0952607714737033, 2.0952607714737033, 2.0952607714737033,
                            2.0952607714737033, 2.0952607714737033]
    },
    'BDCMR': {
        # BatchNorm->Dropout->Conv2d->MaxPool->ReLU, no LRN or CrossMapLRN2d
        # Batch 128 on Alienware 51m NVIDIA 2080Ti
        'RMS_Errors': [2.2792, 2.1564, 2.1008, 2.0834, 2.0686, 2.0394, 2.0582, 2.0767, 2.0929, 2.0364, 2.0401, 2.0394,
                       2.0723, 2.0292, 2.0341, 2.0799, 2.0749, 2.0886, 2.0446, 2.1027, 2.0727, 2.1012, 2.0383, 2.0473,
                       2.0619, 2.0223, 2.0581, 2.0354, 2.0375, 2.0400],
        'Best_RMS_Errors': [2.2792, 2.1564, 2.1008, 2.0834, 2.0686, 2.0394, 2.0394, 2.0394, 2.0394, 2.0364, 2.0364,
                            2.0364, 2.0364, 2.0292, 2.0292, 2.0292, 2.0292, 2.0292, 2.0292, 2.0292, 2.0292, 2.0292,
                            2.0292, 2.0292, 2.0292, 2.0223, 2.0223, 2.0223, 2.0223, 2.0223],
    },
    'BDCMR-crop': {
        # BDCMR, don't do mean for face/eyes, add random crop from 240->224
        # Batch 128 on Alienware 51m NVIDIA 2080Ti
        'RMS_Errors': [2.293592200122107, 2.23778991227833, 2.111237920557162, 2.0503367604228098, 2.1053897357740574,
                      2.077477353582786, 2.0204562945901428, 2.038922714529698, 2.06019775089836, 2.0420013571074125,
                      2.015661977006031, 2.0518391786202055, 2.068389495652076, 2.015549033491421, 2.0373749134599883,
                      2.0211977435024138, 2.0409215894094954, 2.0152629176321333, 2.0091810416277416,
                      2.0336313383003652, 1.9962893532520183, 1.9812626102745092, 2.0461399667761673,
                      2.0024401214191716, 1.9876474390433951, 2.014105341076931, 1.97707222670457, 2.003801866882827,
                      2.008408210320848, 2.004988826483628],
        'Best_RMS_Errors': [2.293592200122107, 2.23778991227833, 2.111237920557162, 2.0503367604228098,
                            2.0503367604228098, 2.0503367604228098, 2.0204562945901428, 2.0204562945901428,
                            2.0204562945901428, 2.0204562945901428, 2.015661977006031, 2.015661977006031,
                            2.015661977006031, 2.015549033491421, 2.015549033491421, 2.015549033491421,
                            2.015549033491421, 2.0152629176321333, 2.0091810416277416, 2.0091810416277416,
                            1.9962893532520183, 1.9812626102745092, 1.9812626102745092, 1.9812626102745092,
                            1.9812626102745092, 1.9812626102745092, 1.97707222670457, 1.97707222670457,
                            1.97707222670457, 1.97707222670457]
    },
    'BDCMR-HSM': {
        # BDCMR-crop, plus turn on Hard Sample Mining
        # Batch 128 on Alienware 51m NVIDIA 2080Ti
        'RMS_Errors': [2.331934588406017, 2.2223023436738085, 2.2591254915418286, 2.30623818617502, 2.221061948232388,
                       2.2014403343521356, 2.2300259151471566, 2.113216987454659, 2.1317593734538547,
                       2.1482830236803068, 2.1804695212768883, 2.1530267465331048, 2.1622830679854372,
                       2.1102463601977535, 2.143904024206237, 2.0917060714483102, 2.0678742538944697, 2.117999564102214,
                       2.1636283758108297, 2.1741851597242734, 2.11031271854983, 2.1802302933893674, 2.1296453013865837,
                       2.089823168367849, 2.0939350923757547, 2.115415565391956, 2.1142792543309925, 2.135180254886835,
                       2.14331327151675, 2.122247109544622],
        'Best_RMS_Errors': [2.331934588406017, 2.2223023436738085, 2.2223023436738085, 2.2223023436738085,
                            2.221061948232388, 2.2014403343521356, 2.2014403343521356, 2.113216987454659,
                            2.113216987454659, 2.113216987454659, 2.113216987454659, 2.113216987454659,
                            2.113216987454659, 2.1102463601977535, 2.1102463601977535, 2.0917060714483102,
                            2.0678742538944697, 2.0678742538944697, 2.0678742538944697, 2.0678742538944697,
                            2.0678742538944697, 2.0678742538944697, 2.0678742538944697, 2.0678742538944697,
                            2.0678742538944697, 2.0678742538944697, 2.0678742538944697, 2.0678742538944697,
                            2.0678742538944697, 2.0678742538944697]
    },
    'BDCMR-ADV': {
        # BDCMR-crop, plus turn on adversarial attack - specifically Fast Gradient Sign Attack (FGSA)
        # Batch 128 on Alienware 51m NVIDIA 2080Ti
        'RMS_Errors': [3.118429561293037, 2.8755794966068535, 3.1696789808215593, 2.8034747481586635,
                       2.8611501258896275, 2.50125332359668, 2.7022982818613777, 2.6738781996036267, 2.620483103003691,
                       2.590432259053613, 2.4208231254800823, 2.491220467573216, 2.568318672058321, 2.7575006834836957,
                       2.59086785483216, 2.57945299100459, 2.502373034663864, 2.4565531836473147, 2.4146290859922757,
                       2.4658071307905103, 2.387331842237779, 2.3507533910375518, 2.4609118867257176,
                       2.3382935154574302, 2.4258736207329674, 2.424054445447582, 2.327015259735682, 2.5391822532430623,
                       2.3146083498931045, 2.3213220215356825],
        'Best_RMS_Errors': [3.118429561293037, 2.8755794966068535, 2.8755794966068535, 2.8034747481586635,
                            2.8034747481586635, 2.50125332359668, 2.50125332359668, 2.50125332359668, 2.50125332359668,
                            2.50125332359668, 2.4208231254800823, 2.4208231254800823, 2.4208231254800823,
                            2.4208231254800823, 2.4208231254800823, 2.4208231254800823, 2.4208231254800823,
                            2.4208231254800823, 2.4146290859922757, 2.4146290859922757, 2.387331842237779,
                            2.3507533910375518, 2.3507533910375518, 2.3382935154574302, 2.3382935154574302,
                            2.3382935154574302, 2.327015259735682, 2.327015259735682, 2.3146083498931045,
                            2.3146083498931045]
    },
    'BDCMR-227': {
        # BDCMR-227, input images are 227x227. No mean applied, required since the means were calculated for
        # images 224x224. 227x227 picked to make the math for AlexNet work more cleanly. Slightly larger network overall
        # due to larger downstream convolutions (ie: 27x27 instead of 26x26, 13x13 instead of 12x12) and thus larger
        # fully connected layers (ie: 10,816 vs 9,216.
        # Batch 128 on Alienware 51m NVIDIA 2080Ti
        'RMS_Errors': [2.2848872596815117, 2.2313569920609764, 2.1330111509693817, 2.1532837128591122,
                       2.089535972618303, 2.1353527292919865, 2.080171441101916, 2.055652209168404, 2.068506752987716,
                       2.0746394579732828, 2.0729753325269296, 2.0677839394295545, 2.121268363855153,
                       2.0411622707364376, 2.060806975047221, 2.0877841929260645, 2.0717681669274994, 2.083143561144523,
                       2.074397409435239, 2.0694584253256, 2.065985556040022, 2.0571484905004342, 2.0748454249137667,
                       2.0920052580971324, 2.0692431326111294, 2.060493913238772, 2.0771870811272564, 2.044604203295371,
                       2.071119549776294, 2.101963849603208],
        'Best_RMS_Errors': [2.2848872596815117, 2.2313569920609764, 2.1330111509693817, 2.1330111509693817,
                            2.089535972618303, 2.089535972618303, 2.080171441101916, 2.055652209168404,
                            2.055652209168404, 2.055652209168404, 2.055652209168404, 2.055652209168404,
                            2.055652209168404, 2.0411622707364376, 2.0411622707364376, 2.0411622707364376,
                            2.0411622707364376, 2.0411622707364376, 2.0411622707364376, 2.0411622707364376,
                            2.0411622707364376, 2.0411622707364376, 2.0411622707364376, 2.0411622707364376,
                            2.0411622707364376, 2.0411622707364376, 2.0411622707364376, 2.0411622707364376,
                            2.0411622707364376, 2.0411622707364376]
    },
    'BDCMR-227-crop': {
        # BDCMR-227, add random crop from 240->227.
        # Batch 128 on Alienware 51m NVIDIA 2080Ti
        'RMS_Errors': [2.314591295412906, 2.2165885619693904, 2.15799731368095, 2.042023872735922, 2.0753487422471086,
                       2.082726178557262, 2.028972923507126, 2.056609269766176, 2.030756619253973, 2.0462504959619516,
                       2.0099763852923047, 2.0382958848923747, 2.017444107399423, 2.0239129554527913, 2.002983872202634,
                       2.0394760656453013, 2.0285265246252124, 2.0256752422053137, 2.0295476109530997,
                       2.113513815571337, 2.0321694637034353, 1.9980751998326725, 1.991890024930419, 2.020401748203157,
                       2.0518136022551072, 2.070038655113677, 2.021048586177762, 2.010420237664978, 2.048875621957272,
                       2.0454114986732654],
        'Best_RMS_Errors': [2.314591295412906, 2.2165885619693904, 2.15799731368095, 2.042023872735922,
                            2.042023872735922, 2.042023872735922, 2.028972923507126, 2.028972923507126,
                            2.028972923507126, 2.028972923507126, 2.0099763852923047, 2.0099763852923047,
                            2.0099763852923047, 2.0099763852923047, 2.002983872202634, 2.002983872202634,
                            2.002983872202634, 2.002983872202634, 2.002983872202634, 2.002983872202634,
                            2.002983872202634, 1.9980751998326725, 1.991890024930419, 1.991890024930419,
                            1.991890024930419, 1.991890024930419, 1.991890024930419, 1.991890024930419,
                            1.991890024930419, 1.991890024930419]
    },
    'BDCMR-CLR': {
        # BDCMR-crop, add CLR in triangular shape using 3E-3 as upper bound, 3E-3/6 as lower bound
        # Batch 128 on Alienware 51m NVIDIA 2080Ti
        'RMS_Errors': [2.37913497101964, 2.256394443531203, 2.2498499778267838, 2.1023626527613093, 2.0570811328746843,
                       2.0420069148737707, 2.027525443940974, 2.019467522108723, 2.024614392564693, 2.0622427220783863,
                       2.075644106278102, 2.1231898370010387, 2.079519994738283, 2.0054482663327122, 1.9801020802470286,
                       1.9761306966962475, 2.006571741880149, 1.9715972641609463, 2.0903384909023752, 2.104521881082993,
                       2.055211592810861, 2.0141416943466095, 1.9869592400469391, 1.9842890210013784, 1.991951227508781,
                       2.0294997850840093, 2.019113934400824, 2.121046531192392, 2.0463625383922217,
                       2.0002399273021316],
        'Best_RMS_Errors': [2.37913497101964, 2.256394443531203, 2.2498499778267838, 2.1023626527613093,
                            2.0570811328746843, 2.0420069148737707, 2.027525443940974, 2.019467522108723,
                            2.019467522108723, 2.019467522108723, 2.019467522108723, 2.019467522108723,
                            2.019467522108723, 2.0054482663327122, 1.9801020802470286, 1.9761306966962475,
                            1.9761306966962475, 1.9715972641609463, 1.9715972641609463, 1.9715972641609463,
                            1.9715972641609463, 1.9715972641609463, 1.9715972641609463, 1.9715972641609463,
                            1.9715972641609463, 1.9715972641609463, 1.9715972641609463, 1.9715972641609463,
                            1.9715972641609463, 1.9715972641609463]
    },
    'BDCMR-jitter-normalize': {
        # BDCMR-CLR, plus jitter and ImageNet normalization
        # Batch 128 on Alienware 51m NVIDIA 2080Ti
        'RMS_Errors': [2.2764698737567115, 2.199612938740467, 2.1317177972299746, 2.17347160084743, 2.016479894725284,
                       2.026111918034454, 1.990147227509838, 1.958640482799271, 1.9673514158676035, 1.9964695596342208,
                       1.9808058446898265, 2.0627382295758596, 2.0073481304178964, 1.9820244700618455,
                       1.941597229663205, 1.9375486504414294, 1.9542620590251008, 1.9879782461526816, 1.986629602808077,
                       1.9878866461193954, 2.008785710081448, 1.9449975675917666, 1.9549336130376942,
                       1.9280841124482657, 1.9395810865121315, 1.9612372202126067, 1.9579638661998346,
                       1.9969564977875474, 1.9887267873041887, 1.9472573110379063],
        'Best_RMS_Errors': [2.2764698737567115, 2.199612938740467, 2.1317177972299746, 2.1317177972299746,
                            2.016479894725284, 2.016479894725284, 1.990147227509838, 1.958640482799271,
                            1.958640482799271, 1.958640482799271, 1.958640482799271, 1.958640482799271,
                            1.958640482799271, 1.958640482799271, 1.941597229663205, 1.9375486504414294,
                            1.9375486504414294, 1.9375486504414294, 1.9375486504414294, 1.9375486504414294,
                            1.9375486504414294, 1.9375486504414294, 1.9375486504414294, 1.9280841124482657,
                            1.9280841124482657, 1.9280841124482657, 1.9280841124482657, 1.9280841124482657,
                            1.9280841124482657, 1.9280841124482657]
    },
}

# Make a data frame
rms_object = {'x': range(1, 31)}
for key in All_RMS_Errors.keys():
    rms_object[key] = np.array((All_RMS_Errors[key])['RMS_Errors'])

df_rms = pd.DataFrame(rms_object)

# style
plt.style.use('seaborn-darkgrid')

# create a color palette
palette = plt.get_cmap('Set1')

# multiple line plot
num = 0
for column in df_rms.drop('x', axis=1):
    num += 1
    plt.plot(df_rms['x'], df_rms[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)

# Add legend
plt.legend(loc=2, ncol=2)

# Add titles
plt.title("RMS Errors by Epoch", loc='left', fontsize=12, fontweight=0, color='orange')
plt.xlabel("Epoch")
plt.ylabel("RMS Error")
plt.show()

best_rms_object = {'x': range(1, 31)}
for key in All_RMS_Errors.keys():
    best_rms_object[key] = np.array((All_RMS_Errors[key])['Best_RMS_Errors'])

# Make a data frame
df_best_rms = pd.DataFrame(best_rms_object)

# style
plt.style.use('seaborn-darkgrid')

# create a color palette
palette = plt.get_cmap('Set1')

# multiple line plot
num = 0
for column in df_best_rms.drop('x', axis=1):
    num += 1
    plt.plot(df_best_rms['x'], df_best_rms[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)

# Add legend
plt.legend(loc=2, ncol=2)

# Add titles
plt.title("Best RMS Errors by Epoch", loc='left', fontsize=12, fontweight=0, color='orange')
plt.xlabel("Epoch")
plt.ylabel("RMS Error")
plt.show()

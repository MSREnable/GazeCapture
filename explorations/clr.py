import matplotlib.pyplot as plt

lr_find_loss = [35.596065521240234, 35.12154121398926, 34.67078832626343, 34.230623925209045, 33.859136464452746,
                33.851094039423465, 33.81221365385854, 33.454448659031826, 33.186639365423495, 33.47238224151267,
                33.39534105988747, 33.31172446236795, 32.94944853816923, 32.69801836284768, 33.050733411502165,
                32.85672707158501, 32.63342196800576, 32.36717693314307, 32.29594447503572, 32.09250608459692,
                32.137695195345096, 32.12728813485762, 32.01479694283642, 31.84500983541017, 31.971785909191418,
                31.984802067223058, 31.708944246118982, 31.580305124022992, 31.478466438714175, 31.62831631013784,
                31.733824093874112, 31.51465999260448, 31.73569308336732, 31.855684285217016, 31.631056504061632,
                31.471542806208404, 31.243340345463412, 31.1287342107586, 31.369116233436003, 31.293552973186323,
                31.248392174044827, 31.011410683323543, 31.027110595934708, 31.07216200278348, 31.129569794673113,
                30.959328830879397, 30.818194321403297, 30.905200559922974, 30.740791705327702, 30.485497649236123,
                30.455009483235497, 30.421458456705558, 30.388193475307045, 29.833423846707706, 30.03097670680152,
                29.995620879884296, 29.891240932234076, 29.728757437868467, 29.416048413478702, 29.599243032599688,
                29.532330388110815, 29.528736711112497, 29.71641425543968, 29.741996626468964, 29.420277654520515,
                29.227563771031548, 28.991014493307606, 28.922455513637342, 28.595304467234037, 28.737044408972434,
                28.549168991349738, 28.49481034570681, 28.198483696371905, 28.12602641905453, 28.108115543811028,
                27.721767708057243, 27.56744506972469, 27.270132939682057, 26.85208146237569, 26.427659953038642,
                26.13785139538915, 25.546253615887633, 25.085293493229237, 24.85538384862759, 24.63546812604484,
                24.504571623276536, 24.45743809673917, 24.128728942146353, 23.979398703840303, 23.981630589021822,
                24.000771105469166, 24.347168383173003, 24.191376507593453, 24.503371307549475, 26.45109912736487,
                26.536568768164596, 26.607034088148698, 26.53554764481914, 26.752995383122613, 49.32044815302898]
lr_find_lr = [3.507518739525677e-06, 3.8904514499428085e-06, 4.315190768277653e-06, 4.78630092322639e-06,
              5.308844442309879e-06, 5.888436553555883e-06, 6.531305526474725e-06, 7.244359600749899e-06,
              8.035261221856182e-06, 8.912509381337446e-06, 9.88553094656939e-06, 1.0964781961431848e-05,
              1.2161860006463673e-05, 1.3489628825916545e-05, 1.4962356560944312e-05, 1.6595869074375605e-05,
              1.8407720014689548e-05, 2.0417379446695308e-05, 2.2646443075930562e-05, 2.5118864315095754e-05,
              2.7861211686297692e-05, 3.090295432513588e-05, 3.427677865464505e-05, 3.8018939632056056e-05,
              4.216965034285821e-05, 4.6773514128719775e-05, 5.188000389289614e-05, 5.75439937337157e-05,
              6.382634861905473e-05, 7.079457843841374e-05, 7.85235634610071e-05, 8.709635899560808e-05,
              9.660508789898131e-05, 0.00010715193052376076, 0.0001188502227437017, 0.00013182567385564074,
              0.00014621771744567177, 0.00016218100973589288, 0.0001798870915128789, 0.00019952623149688766,
              0.00022130947096056375, 0.0002454708915685029, 0.00027227013080779143, 0.00030199517204020174,
              0.000334965439157827, 0.00037153522909717237, 0.0004120975190973298, 0.00045708818961487515,
              0.0005069907082747043, 0.0005623413251903488, 0.0006237348354824186, 0.0006918309709189356,
              0.0007673614893618189, 0.0008511380382023746, 0.0009440608762859243, 0.0010471285480509,
              0.0011614486138403427, 0.0012882495516931334, 0.0014288939585111017, 0.0015848931924611115,
              0.0017579236139586897, 0.001949844599758048, 0.0021627185237270146, 0.002398832919019492,
              0.00266072505979881, 0.002951209226666385, 0.0032734069487883794, 0.0036307805477010097,
              0.0040271703432545845, 0.004466835921509622, 0.004954501908047908, 0.00549540873857623,
              0.006095368972401693, 0.006760829753919817, 0.0074989420933245544, 0.008317637711026702,
              0.009225714271547619, 0.010232929922807523, 0.011350108156723123, 0.012589254117941682,
              0.01396368361055938, 0.015488166189124814, 0.017179083871575872, 0.019054607179632456,
              0.02113489039836644, 0.023442288153199178, 0.026001595631652757, 0.02884031503126598, 0.03198895109691399,
              0.035481338923357544, 0.03935500754557773, 0.04365158322401656, 0.04841723675840988, 0.05370317963702518,
              0.05956621435290092, 0.06606934480075967, 0.07328245331389045, 0.08128305161640965, 0.09015711376059567,
              0.09999999999999994]

min_loss = min(lr_find_loss)
index = lr_find_loss.index(min_loss)
upper_lr = lr_find_lr[index] / 10
lower_lr = upper_lr / 6

print(f'min_loss {min_loss}, lr {lr_find_lr[index]}')
print(f'upper_lr = {upper_lr}')
print(f'lower_lr = {lower_lr}')

plt.ylabel("lr")
plt.xlabel("step")
plt.plot(range(len(lr_find_lr)), lr_find_lr)
plt.show()

plt.ylabel("loss")
plt.xlabel("lr")
plt.xscale("log")
plt.plot(lr_find_lr, lr_find_loss)
plt.xscale('log')
plt.show()
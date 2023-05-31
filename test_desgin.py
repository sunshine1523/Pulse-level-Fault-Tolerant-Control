
import matplotlib.pyplot as plt
import numpy as np
from qutip import *
import pickle
from mpl_toolkits.mplot3d import Axes3D
### Rabi Oscillation 的模拟

Nstates = 3
ground = basis(Nstates, 0)
# print(ground)
# excited = basis(Nstates, 1)
phi = np.pi
target_U = Qobj([[np.cos(phi / 2), -1j * np.sin(phi / 2), 0], [-1j * np.sin(phi / 2), np.cos(phi / 2), 0], [0, 0, 0]])
# print(target_U)
excited = target_U * ground
# excited = Qobj(excited)
# print(excited)
sigma_gg = ground * ground.dag()   # |g><g|
sigma_ee = excited * excited.dag() # |e><e|

a = destroy(Nstates)
adag = a.dag()

# 单位矩阵
I_q = qeye(Nstates)

t = np.linspace(-15, 15, 1000) #时间演化范围

psi0 = ground #基态
psi1 = excited # 第一激发态
# 泄露能级
leakage = basis(Nstates,2)

fa = 6 #in GHz
wa = fa * 2*np.pi  #Qubit的频率
alpha = -0.3 * 2 * np.pi  #Qubit的非谐项，单位 GHz

H0 = wa*adag*a #哈密顿量
Hnonlin = 0.5*adag*adag*a*a #非谐项哈密顿量

def anharmonicity_coeff(t, args): #用来返回非谐项
    return args['alpha']

leak = basis(Nstates, 2)
sigma_ff = leak * leak.dag()




## Pulse
def H1_coeff_gauss(t, args):
    omega=args['omega']
    wd=args['w_drive']
    tau=args['tau']
    amp = 0.5 * omega * np.exp(-(t-10)**2/(tau**2)) * np.exp(1j*wd*t)
    return amp

def H2_coeff_gauss(t, args):
    omega=args['omega']
    wd=args['w_drive']
    tau=args['tau']
    amp = 0.5 * omega * np.exp(-(t-10)**2/(tau**2)) * np.exp(-1j*wd*t)
    return amp

def omega(tau):
    return np.pi/(2*tau)

def omega_gauss(tau,strength):
    return omega(tau)*strength


### Transmon
H=[H0,[Hnonlin,anharmonicity_coeff],[a,H1_coeff_gauss],[adag,H2_coeff_gauss]]

# 拟合部分
from scipy.optimize import curve_fit

def fit_function(x_values, y_values, function, init_params):
    fitparams, conv = curve_fit(function, x_values, y_values, init_params, maxfev=50000)
    y_fit = function(x_values, *fitparams)

    return fitparams, y_fit

def lorentzian(x,A,f_q,B):
    return A  * (B**2 / ((x - f_q)**2 + B**2))

## time
tau=4
strengthList = np.linspace(0.6,1.5,5)
opts = Options(rhs_reuse=True)
excitedList = []
freqList = []
leakageList = []
#
# f_drive_list = np.linspace(fa - 0.05, fa + 0.05, num=50) # 50MHz
# for item in strengthList:
#     strength = np.sqrt(np.pi) / 2 * item
#     print("Driving Strength ", item)
#     g_list = []
#     e_list = []
#     f_list = []
#     for f in f_drive_list:
#         t = np.linspace(0, 20, 10000)
#         output = mesolve(H, psi0, t, [], [sigma_gg, sigma_ee, sigma_ff],
#                          args={'w_drive': 2 * np.pi * f, 'alpha': alpha, 'omega': omega_gauss(tau, strength),
#                                'tau': tau}, options=opts)
#         g_list.append(output.expect[0][-1])
#         e_list.append(output.expect[1][-1])
#         f_list.append(output.expect[2][-1])
#
#     excitedList.append(e_list)
#     fit_params, y_fit = fit_function(f_drive_list, e_list, lorentzian,
#                                      [1, 6, 1])  # initial parameters for curve_fit                                )
#     print("Stark Frequency", fit_params[1])
#     freqList.append(fit_params[1])
#
# # 提取最佳的振幅和频移，jiu我们在最佳
# opt_strength = np.sqrt(np.pi) / 2 * strengthList[np.where(excitedList == np.max(excitedList))[0][0]]
# opt_freq = 2 * np.pi * f_drive_list[np.where(excitedList == np.max(excitedList))[1][0]]  # drive frequency for your best pulse
# print(opt_strength)
# print(opt_freq)
# opt_strength = 1
# opt_freq = 6
opt_freq = 6.00510214325
opt_strength = 1.1520950030885853

t = np.linspace(0,20,10000)
output = mesolve(H, psi1, t, [], [sigma_gg,sigma_ee,sigma_ff],args={'w_drive':2 * np.pi * opt_freq,'alpha':alpha,'omega':omega_gauss(tau,opt_strength),'tau':tau}, options = opts)
print("Final population in e = ",output.expect[0][-1])

# 最后我们的保真度可以达到 g =  0.9996014121133802
#
# opt_freq = 37.73116891097129
# opt_strength = 1.1520950030885853
# 绘制 g 和 e 的布局数

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(t, output.expect[0], label="g", linewidth=3)
ax.plot(t, output.expect[1], label="e", linewidth=3)
ax.plot(t, output.expect[2], label="leakage", linewidth=3)
plt.tick_params(width=2.5, labelsize=12)
ax.legend()
ax.set_xlabel('Time (ns)', fontdict={'size': 15})
ax.set_ylabel('Occupation probability', fontdict={'size': 15})
# ax.set_title('Rabi pi pulse')
fname = 'no-noise_Population_result_' +  '.png'
plt.savefig(fname, dpi=1080)
#
# fig, ax = plt.subplots(figsize=(8,5))
# t = np.linspace(0,20,2000)
# ax.plot(t, np.real(H1_coeff_gauss(t,{'w_drive':opt_freq,'omega':omega_gauss(tau,opt_strength),'tau':tau})) + np.real(H2_coeff_gauss(t,{'w_drive':opt_freq,'omega':omega_gauss(tau,opt_strength),'tau':tau})), label="real")
# ax.legend()
# ax.set_xlabel('Time(ns)')
# ax.set_ylabel('Amplitude')
# ax.set_title('Applied pulse')
# fname = 'pulse_result_' +  '.png'
# plt.savefig(fname, dpi=1080)



def change_strength_gauss(opt_strength, opt_freq):
    fide_change = []
    tau = 4
    t = np.linspace(0, 20, 1000)
    strengthList = np.linspace(0.6, 1.5, 10)
    # tau_list = np.linspace(2, 6, 50)
    for strength_ in strengthList:
        output = mesolve(H, psi1, t, [], [sigma_gg, sigma_ee, sigma_ff],
                     args={'w_drive': opt_freq, 'alpha': alpha, 'omega': omega_gauss(tau, strength_), 'tau': tau},
                     options=opts)
        fide_change.append(output.expect[0][-1])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(strengthList, fide_change,
            label="strength")
    ax.legend()
    ax.set_xlabel('Omega_t')
    ax.set_ylabel('Fidelity')
    ax.set_title('Generalization')
    fname = 'generalization_result_strength' + '.png'
    plt.savefig(fname, dpi=1080)

def change_tau(opt_strength, opt_freq):
    fide_change = []

    opt_strength = 1
    f_drive = 6
    t = np.linspace(0, 20, 1000)
    tau_list = np.linspace(-2 + 4, 2 + 4, num=25)
    for tau in tau_list:
        output = mesolve(H, psi1, t, [], [sigma_gg, sigma_ee, sigma_ff],
                         args={'w_drive': 2 * np.pi * f_drive, 'alpha': alpha, 'omega': omega_gauss(tau, opt_strength),
                               'tau': tau},
                         options=opts)
        fide_change.append(output.expect[0][-1])
    z_pi = fide_change
    fide_change = []
    opt_strength = 1.1299393299522662
    opt_freq = 37.73116891097129
    t = np.linspace(0, 20, 1000)
    tau_list = np.linspace(-2 + 4, 2 + 4, num=25)
    for tau in tau_list:
        output = mesolve(H, psi1, t, [], [sigma_gg, sigma_ee, sigma_ff],
                         args={'w_drive': opt_freq, 'alpha': alpha, 'omega': omega_gauss(tau, opt_strength),
                               'tau': tau},
                         options=opts)
        fide_change.append(output.expect[0][-1])
    z_model = fide_change
    fide_change = []
    f_list = open("d:/PycharmProjects/SSRL_10.19/SSRL_tau_data_1.pkl", "rb")
    data = pickle.load(f_list)
    z_free = data['fide']
    tau_list_ = np.linspace(-2, 2, num=25)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax = fig.gca()
    ax.spines['top'].set_linewidth(5)
    ax.spines['right'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    plt.tick_params(width=15.5, labelsize=30)
    ax.plot(tau_list_, z_pi,
            label="no-noise", linewidth=9, linestyle='dotted')
    ax.plot(tau_list_, z_model,
            label="model", linewidth=9, linestyle='dashed')
    ax.plot(tau_list_, z_free, label="SSRQ", linewidth=9, linestyle='dashdot')
    ax.legend(prop={'size': 33}, loc='lower right', ncol=3, frameon=False)
    ax.set_xlabel('Omega_t(MHz)', fontdict={'size': 45})
    ax.set_ylabel('Fidelity', fontdict={'size': 45})
    # ax.set_title('Generalization')
    fname = 'all_generalization_result_tau' + '.png'
    plt.savefig(fname, dpi=1080, bbox_inches='tight')

def change_f_drive(opt_strength, opt_freq):
    fide_change = []

    opt_strength = 1
    t = np.linspace(0, 20, 1000)
    f_drive_list = np.linspace(fa - 0.005, fa + 0.01, num=25)  # 20MHz
    for f_drive in f_drive_list:
        output = mesolve(H, psi1, t, [], [sigma_gg, sigma_ee, sigma_ff],
                     args={'w_drive': 2 * np.pi * f_drive, 'alpha': alpha, 'omega': omega_gauss(tau, opt_strength), 'tau': tau},
                     options=opts)
        fide_change.append(output.expect[0][-1])
    z_pi = fide_change
    fide_change = []
    opt_strength = 1.1299393299522662
    t = np.linspace(0, 20, 1000)
    f_drive_list = np.linspace(fa - 0.005, fa + 0.01, num=25)  # 20MHz
    for f_drive in f_drive_list:
        output = mesolve(H, psi1, t, [], [sigma_gg, sigma_ee, sigma_ff],
                         args={'w_drive': 2 * np.pi * f_drive, 'alpha': alpha, 'omega': omega_gauss(tau, opt_strength),
                               'tau': tau},
                         options=opts)
        fide_change.append(output.expect[0][-1])
    z_model = fide_change
    fide_change = []
    f_list = open("d:/PycharmProjects/SSRL_10.19/SSRL_drive_data_1.pkl", "rb")
    data = pickle.load(f_list)
    z_free = data['fide']
    f_drive_list_ = np.linspace(- 0.005 * 1000, 0.01 * 1000, num=25)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax = fig.gca()
    ax.spines['top'].set_linewidth(5)
    ax.spines['right'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    plt.tick_params(width=15.5, labelsize=30)
    ax.plot(f_drive_list_, z_pi,
            label="no-noise", linewidth=9, linestyle='dotted')
    ax.plot(f_drive_list_, z_model,
            label="model", linewidth=9, linestyle='dashed')
    ax.plot(f_drive_list_, z_free, label="SSRQ", linewidth=9, linestyle='dashdot')
    ax.legend(prop={'size': 33}, loc='lower right', ncol=3, frameon=False)
    ax.set_xlabel('Detuning(MHz)', fontdict={'size': 45})
    ax.set_ylabel('Fidelity', fontdict={'size': 45})
    # ax.set_title('Generalization')
    fname = 'all_generalization_result_Detuning' + '.png'
    plt.savefig(fname, dpi=1080, bbox_inches='tight')

def change_alpha_anharmonicity(opt_strength, opt_freq):
    fide_change = []
    tau = 4
    opt_strength = 1
    f_drive = 6
    t = np.linspace(0, 20, 1000)
    alpha_list = np.linspace(-0.4, -0.2, num=25)
    for alpha_ in alpha_list:
        output = mesolve(H, psi1, t, [], [sigma_gg, sigma_ee, sigma_ff],
                         args={'w_drive': 2 * np.pi * f_drive, 'alpha': alpha_ * 2 * np.pi, 'omega': omega_gauss(tau, opt_strength),
                               'tau': tau},
                         options=opts)
        fide_change.append(output.expect[0][-1])
    z_pi = fide_change
    fide_change = []
    opt_strength = 1.1299393299522662
    opt_freq = 37.73116891097129
    t = np.linspace(0, 20, 1000)
    alpha_list = np.linspace(-0.4, -0.2, num=25)
    for alpha_ in alpha_list:
        output = mesolve(H, psi1, t, [], [sigma_gg, sigma_ee, sigma_ff],
                         args={'w_drive': opt_freq, 'alpha': alpha_ * 2 * np.pi, 'omega': omega_gauss(tau, opt_strength),
                               'tau': tau},
                         options=opts)
        fide_change.append(output.expect[0][-1])
    z_model = fide_change
    fide_change = []
    f_list = open("d:/PycharmProjects/SSRL_10.19/SSRL_alpha_data_1.pkl", "rb")
    data = pickle.load(f_list)
    z_free = data['fide']
    alpha_list_ = np.linspace(-0.4 * 1000, -0.2 * 1000, num=25)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax = fig.gca()
    ax.spines['top'].set_linewidth(5)
    ax.spines['right'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    plt.tick_params(width=15.5, labelsize=30)
    ax.plot(alpha_list_, z_pi,
            label="no-noise", linewidth=9, linestyle='dotted')
    ax.plot(alpha_list_, z_model,
            label="model", linewidth=9, linestyle='dashed')
    ax.plot(alpha_list_, z_free, label="SSRQ", linewidth=9, linestyle='dashdot')
    ax.legend(prop={'size': 33}, loc='center right', ncol=3, frameon=False)
    ax.set_xlabel('Anharmonicity(MHz)', fontdict={'size': 45})
    ax.set_ylabel('Fidelity', fontdict={'size': 45})
    # ax.set_title('Generalization')
    fname = 'all_generalization_result_alpha' + '.png'
    plt.savefig(fname, dpi=1080, bbox_inches='tight')

# opt_freq = 37.73116891097129
# opt_strength = 1.1520950030885853
opt_strength = 1.1299393299522662
# opt_strength = 1
opt_freq = 37.73116891097129
# change_omega_gauss(opt_strength, opt_freq)
# change_tau(opt_strength, opt_freq)
def change_omega_and_fa_gauss(opt_strength, opt_freq):
    fide_change = []
    x = []
    y = []
    num_one = 0
    num_two = 0
    num_three = 0
    # t = np.linspace(0, 20, 1000)
    # f_drive_list = np.linspace(fa - 0.005, fa + 0.01, num=25)  # 5MHz
    # for f_drive in f_drive_list:
    #     tau_list = np.linspace(2, 6, num=25)
    #     for tau_ in tau_list:
    #         output = mesolve(H, psi1, t, [], [sigma_gg, sigma_ee, sigma_ff],
    #                      args={'w_drive': 2 * np.pi * f_drive, 'alpha': alpha, 'omega': omega_gauss(tau_, opt_strength), 'tau': tau_},
    #                      options=opts)
    #         fide_change.append(output.expect[0][-1])
    #         x.append((f_drive - fa) * 1000)
    #         y.append(tau_ - 4)
    #         if output.expect[0][-1] >= 0.95:
    #             num_one += 1
    #         if output.expect[0][-1] >= 0.99:
    #             num_two += 1
    #         if output.expect[0][-1] >= 0.999:
    #             num_three += 1
    #
    # data = {"Detuning": x, "Omega_t": y, "fide": fide_change}
    #
    # print(fide_change)
    # print(num_one)
    # print(num_two)
    # print(num_three)
    # with open("d:/PycharmProjects/SSRL_10.19/3D_data_1.pkl", "wb") as f:
    #     pickle.dump(data, f)
    f_list = open("d:/PycharmProjects/SSRL_10.19/3D_data.pkl", "rb")
    data = pickle.load(f_list)
    # z = np.array(data['fide'])
    # z = z.reshape(25, 25)
    # fig1 = plt.figure()
    # fig, ax1 = plt.subplots(figsize=(15, 9))
    # # ax1 = fig.gca()
    # # ax1.spines['top'].set_linewidth(5)
    # # ax1.spines['right'].set_linewidth(5)
    # # ax1.spines['bottom'].set_linewidth(5)
    # # ax1.spines['left'].set_linewidth(5)
    # plt.tick_params(width=15.5, labelsize=30)
    # f_drive_list_ = np.linspace(- 0.005 * 1000, 0.01 * 1000, num=25)
    # tau_list_ = np.linspace(-2, 2, num=25)
    # surf_ = ax1.contourf(f_drive_list_, tau_list_, z, cmap=plt.cm.Spectral)
    # ax1.set_xlabel('Detuning(MHz)', fontdict={'size': 45})
    # ax1.set_ylabel('Omega_t(MHz)', fontdict={'size': 45})
    #
    # fig1.colorbar(surf_, shrink=1)
    #
    # fname = 'generalization_result_2D' + '.png'
    # plt.savefig(fname, dpi=1080, bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_trisurf(data['Detuning'], data['Omega_t'], data['fide'], cmap=plt.cm.Spectral, linewidth=0.2)
    ax = fig.gca()
    ax.spines['top'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    # plt.tick_params(width=15.5, labelsize=30)
    plt.gcf().subplots_adjust(left=0.1, right=None, top=0.91, bottom=0.09)
    ax.view_init(10, 50)
    # ax.legend()
    # ax.set_zlim3d(0.9, 1)
    ax.set_xlabel('Detuning(MHz)', fontdict={'size': 15})
    ax.set_ylabel('Omega_t(MHz)', fontdict={'size': 15})
    ax.set_zlabel('Fidelity', fontdict={'size': 15})
    # ax.set_title('Generalization')
    plt.rcParams['font.size'] = 12
    fig.colorbar(surf, shrink=0.6, ticks=[0.9, 0.95, 0.99, 0.999])
    fname = 'generalization_result_3D' + '.png'
    plt.savefig(fname, bbox_inches='tight')
    f_list.close()

if __name__ == '__main__':

    change_omega_and_fa_gauss(opt_strength, opt_freq)

# change_tau(opt_strength, opt_freq)
#
# change_f_drive(opt_strength, opt_freq)
#
# change_alpha_anharmonicity(opt_strength, opt_freq)
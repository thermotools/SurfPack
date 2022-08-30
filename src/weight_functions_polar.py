#!/usr/bin/env python3
# This file is a proto-typing environment for spherical
import numpy as np
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utility import weighted_densities_1D, differentials_1D, \
    weighted_densities_pc_saft_1D, differentials_pc_saft_1D
import matplotlib.pyplot as plt
import pyfftw as fftw
from scipy.fft import dct, idct, dst, idst, fft, ifft
from scipy.special import jv
from weight_functions import Weights
from scipy.special import spherical_jn

class polar(object):

    def __init__(self,
                 R,
                 domain_size=15.0,
                 n_grid=1024):
        """Set up grid according tO Xi eta al. 2020
        An Efficient Algorithm for Molecular Density Functional Theory in Cylindrical Geometry:
        Application to Interfacial Statistical Associating Fluid Theory (iSAFT)
        DOI: 10.1021/acs.iecr.9b06895
        """
        self.R = R
        self.n_grid = n_grid
        alpha = 0.002
        for _ in range(21):
            alpha = -np.log(1.0 - np.exp(-alpha)) / (n_grid - 1)
        self.alpha = alpha
        #print("alpha",alpha)
        self.x0 = 0.5 * (np.exp(-alpha * n_grid) + np.exp(-alpha * (n_grid - 1)))
        # Setting the grid
        self.z = np.zeros(n_grid)
        for i in range(n_grid):
            self.z[i] = domain_size*self.x0*np.exp(alpha*i)
        # Setting the edge grid
        self.z_edge = np.zeros(n_grid+1)
        for i in range(1,n_grid+1):
            self.z_edge[i] = domain_size*np.exp(-alpha*(n_grid-i))
        # End correction factor
        k0 = np.exp(2*alpha)*(2*np.exp(alpha) + np.exp(2*alpha) - 1)/ \
            (1 + np.exp(alpha))**2/(np.exp(2*alpha) - 1)
        k0v = np.exp(2*alpha)*(2*np.exp(alpha) + np.exp(2*alpha) - 5.0/3.0)/ \
            (1 + np.exp(alpha))**2/(np.exp(2*alpha) - 1)
        self.k0 = k0
        self.kv0 = k0v
        # print("k0", k0)
        # print("k0v", k0v)
        # Hankel paramaters
        self.b = domain_size
        #fac = 1.0/(2*self.x0*(np.exp(alpha*(n_grid-1)) - np.exp(alpha*(n_grid-2))))
        #self.lam = int(0.5*fac/self.b)
        #self.gamma = self.lam*self.b
        # Defining integration weights
        self.integration_weights = np.zeros(n_grid)
        #self.integration_weights[0] = k0 * np.exp(2 * alpha)
        #self.integration_weights[1] = (np.exp(2 * alpha) - k0) * np.exp(2 * alpha)
        for i in range(1,n_grid):
            self.integration_weights[i] = np.exp(2 * alpha * i) * (np.exp(2 * alpha) - 1.0)
        self.integration_weights *= np.exp(-2 * alpha * n_grid) * np.pi * domain_size**2
        self.integration_weights[0] = np.pi*self.z_edge[1]**2

        # for i in range(0,n_grid):
        #     print(i, self.integration_weights[i], np.pi*(self.z_edge[i+1]**2 - self.z_edge[i]**2))
        # sys.exit()
        # print("z",self.z)
        # print("z_edge",self.z_edge)
        # print("gamma", np.exp(alpha * (n_grid-1)))
        # print("l", domain_size)
        # print("iv",self.integration_weights)
        # print("sum_iv",sum(self.integration_weights), np.pi * domain_size**2)
        self.setup_transforms()
        self.calc_lanczos()
        self.analytical_fourier_weigths()

    def tests(self):
        """
        Code testing

        """

        rho = np.array([0.000304946636852787, 0.000304946636852787, 0.000304946636852787,
                        0.000304946636852787, 0.000304946636852787, 0.000304946636852787,
                        0.000304946636852787, 0.000304946636852787, 0.000304946636852787,
                        0.000304946636852787, 0.000304946636852787, 0.000304946636852787,
                        0.000304946636852787, 0.000304946636852787,
                        0.0000000000000000000000000588165779466921,
                        0.0000000000000000000000000588165779466921])

        rho_inf = np.ones(np.shape(rho)[0])*rho[-1]
        rho_delta = rho - rho_inf
        frho_delta = self.transform(rho_delta, scalar=True, forward=True)
        f = frho_delta*self.fw_disp
        n_disp = self.transform(f, scalar=True, forward=False)
        n_disp_inf = rho_inf*self.fw_disp_inf
        n_disp += n_disp_inf

    def get_feos(self):
        """
        Code testing

        """

        rho = np.array([0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0.0006472293155749268, 0, 0, 0, 0, 0])

        n2 = np.array([0.018417397591284152, 0.018422953543730067, 0.018429179581464596, 0.01843616432464142, 0.018444009137420326, 0.01845283001420872, 0.018462759718497264, 0.018473950193000608, 0.018486575254436702, 0.018500833577269123, 0.018516951956733203, 0.018535188820724576, 0.018555837930447367, 0.018579232168375245, 0.018605747255773915, 0.01863580516688744, 0.018669876908493, 0.01870848420712374, 0.018752199487109517, 0.01880164332663528, 0.018857478343971595, 0.018920398193041348, 0.018991110043643993, 0.019070308603560487, 0.01915863943840087, 0.01925664911224169, 0.019364719588472665, 0.01948298451463705, 0.019611225634120235, 0.019748748844137968, 0.01989424163554754, 0.02004561713837433, 0.02019985511228906, 0.020352857276654958, 0.020499343525514246, 0.02063282662614594, 0.020745715128630608, 0.020829605563896982, 0.020875832211410622, 0.020876340384586092, 0.020824929531228806, 0.02071886539541715, 0.020560774714555467, 0.020360601643000024, 0.020137219412974673, 0.019919066378830286, 0.019742952887648383, 0.01965004582151122, 0.01967811325183869, 0.019849580229512536, 0.020155995950181832, 0.020541253504237997, 0.020888220988651854, 0.02101579697350457, 0.020694656101479067, 0.019688382145171924, 0.01782045306760551, 0.015055821729163857, 0.01157062030542234, 0.0077713523824757425, 0.004226840769312046, 0.0015032892903454555, -0.000053679278280213664, -0.000487080655940677])

        n3 = np.array( [0.0091616792706219, 0.009165969056228157, 0.009170779054815918, 0.00917617857891959, 0.00918224699703052, 0.009189075237975358, 0.00919676750041552, 0.009205443183916386, 0.009215239054240196, 0.009226311648963207, 0.009238839919154064, 0.00925302808732034, 0.00926910867942054, 0.009287345657317989, 0.009308037535029467, 0.00933152030442355, 0.00935816992013695, 0.009388403995560849, 0.009422682237906106, 0.009461504997107868, 0.009505409118421581, 0.009554960072214987, 0.009610739091249975, 0.009673323787296767, 0.00974326046762402, 0.009821026165840668, 0.009906978300689174, 0.01000128996983645, 0.010103869299550702, 0.01021426217524253, 0.010331539289091512, 0.010454171018544319, 0.01057989747493489, 0.01070560639367735, 0.010827238528399634, 0.010939748766609633, 0.011037160751837891, 0.011112762073535329, 0.011159493651075523, 0.01117058686400881, 0.011140489535044191, 0.0110660895450284, 0.010948184089012271, 0.010793045725284937, 0.010613800561829081, 0.01043116740409134, 0.010272937582293157, 0.010171461426814335, 0.010158443908714152, 0.01025666755767905, 0.010468992458779962, 0.010766214234044684, 0.011077013206341286, 0.011284918787089765, 0.011238131454682622, 0.010776965089331651, 0.009779289108999235, 0.008216101742946343, 0.0061987230868942415, 0.003990776022327072, 0.0019599540620361057, 0.0004642538220414395, -0.0002947928533492065, -0.00037891677051720833] )

        n2v = -np.array( [0.00012077364005487438, 0.0001289415006732198, 0.00013781891008347223, 0.00014747918484616176, 0.0001580027793598051, 0.00016947756659559265, 0.0001819989719139769, 0.00019566989455682977, 0.0002106003330402833, 0.00022690660847078805, 0.00024471005335097455, 0.00026413500244522754, 0.0002853058866941944, 0.00030834319137431606, 0.0003333579967150035, 0.0003604447750156039, 0.00038967207637095023, 0.0004210707008425255, 0.0004546189364615224, 0.00049022445161503, 0.000527702483629811, 0.0005667500850481582, 0.0006069164044572969, 0.0006475693268538412, 0.0006878593244191829, 0.0007266821240449237, 0.0007626428380293815, 0.0007940255800943725, 0.0008187743342824348, 0.0008344929553333837, 0.0008384745816236906, 0.0008277732456925962, 0.0007993327060074451, 0.0007501888783387867, 0.0006777617634194081, 0.0005802490913582296, 0.00045712524162412567, 0.00030973320171888863, 0.00014193215840468035, -0.000039272977570933104, -0.0002232414510584846, -0.00039552333615305217, -0.0005382365230567179, -0.0006312673490384607, -0.000654564872015303, -0.0005917144312221947, -0.0004347654793568815, -0.00018991048831748015, 0.00011694788921247854, 0.00043639466823801695, 0.0006957771658139107, 0.0008061775581232415, 0.0006781910137280629, 0.000246745106805428, -0.0004983758585776333, -0.0014846110800379648, -0.0025445043394843737, -0.0034343781891298607, -0.0038934656951432546, -0.003737661213393556, -0.0029580149984735843, -0.0017718383312991925, -0.0005730557933944334, 0.00023257048642507782])

        n_disp = np.array( [0.0006589525937264022, 0.00065894247216662, 0.0006589301868587187, 0.0006589152922270689, 0.000658897254738714, 0.0006588754366458005, 0.0006588490770849481, 0.0006588172702306679, 0.0006587789402143387, 0.000658732812560338, 0.000658677381967751, 0.0006586108763944462, 0.0006585312175991487, 0.0006584359785907226, 0.0006583223388522325, 0.000658187038787364, 0.0006580263356220263, 0.0006578359640346699, 0.0006576111061397705, 0.000657346377165976, 0.000657035835303349, 0.0006566730267754907, 0.0006562510802188642, 0.0006557628678576143, 0.0006552012545798903, 0.0006545594595276594, 0.0006538315576545396, 0.0006530131500142454, 0.0006521022300158883, 0.0006511002666732689, 0.0006500135124744223, 0.0006488545196643796, 0.000647643810524098, 0.0006464115902118695, 0.0006451993105334939, 0.0006440607863361801, 0.0006430624326197541, 0.0006422820349495758, 0.0006418053025271175, 0.0006417193102881412, 0.0006421018603300323, 0.0006430058530762032, 0.0006444380468784983, 0.0006463322083202026, 0.0006485177152829965, 0.0006506862249379525, 0.000652361002048261, 0.0006528756667886459, 0.0006513709332520602, 0.0006468185177817824, 0.0006380797101267133, 0.000624001095044206, 0.0006035412318537946, 0.0005759109320227493, 0.0005406996987459073, 0.0004979580006179335, 0.00044821626395508113, 0.0003924500389909178, 0.00033203933235588637, 0.0002687946040136506, 0.00020509480321984235, 0.0001440748594249741, 0.0000896336810670265, 0.00004592495906563943])
        return rho, n2, n3, n2v, n_disp

    def get_feos_pd(self):
        """
        Code testing

        """

        pd_n2 = np.array([0.0022437653153999275, 0.0022444856343197283, 0.0022452929318547, 0.0022461987304168685, 0.0022472162129410895, 0.0022483604693723483, 0.002249648776346007, 0.002251100912588106, 0.0022527395118881926, 0.0022545904543562097, 0.0022566832948914934, 0.00225905172514013, 0.002261734061411273, 0.002264773745709315, 0.0022682198397869914, 0.002272127482426207, 0.0022765582674280655, 0.0022815804834098565, 0.002287269135827132, 0.0022937056461023427, 0.0023009770919989138, 0.002309174817508001, 0.002318392200338, 0.002328721322644546, 0.0023402482497890723, 0.0023530465892425533, 0.0023671689876090455, 0.0023826362435383297, 0.0023994237898662288, 0.002417445459156893, 0.0024365347306561026, 0.002456424108591096, 0.002476723950538616, 0.0024969029942262947, 0.002516274045012374, 0.0025339897608064755, 0.0025490550959975707, 0.0025603644896145454, 0.0025667728475336662, 0.0025672090455101343, 0.002560838033894483, 0.0025472713523071898, 0.002526814569810266, 0.0025007227680165414, 0.002471411588896812, 0.002442543468291148, 0.0024188815773816294, 0.002405787296851871, 0.0024082458233831146, 0.0024293593557286417, 0.002468372170665262, 0.0025185067763023017, 0.002565193813052393, 0.0025856146917216773, 0.002550690491076437, 0.0024304667134400597, 0.002202937096934343, 0.0018646038111296303, 0.0014389565934706085, 0.0009777962583760258, 0.0005514127976092167, 0.0002276998869138007, 4.636203665112726e-5, 0.0])
        pd_n3 = np.array([0.0006271327276150065, 0.0006273251897412892, 0.0006275408675048333, 0.0006277828321821459, 0.0006280545970081115, 0.0006283601826896182, 0.000628704191729996, 0.0006290918922337024, 0.0006295293116816147, 0.0006300233408645884, 0.000630581847689938, 0.0006312137998723019, 0.0006319293945111572, 0.0006327401911479129, 0.0006336592429713958, 0.000634701218265894, 0.0006358825008147738, 0.0006372212536144705, 0.0006387374247439777, 0.0006404526674169129, 0.0006423901380122318, 0.0006445741262393642, 0.0006470294607404597, 0.000649780621879483, 0.0006528504822184731, 0.0006562585859649123, 0.0006600188742516784, 0.0006641367676468873, 0.0006686055367677941, 0.0006734019344841561, 0.0006784811395790579, 0.0006837711847954946, 0.0006891672261163437, 0.0006945262679206903, 0.0006996632974457421, 0.0007043501949358348, 0.0007083192400084724, 0.0007112734545176223, 0.0007129062703240661, 0.000712932871666429, 0.0007111347393982321, 0.0007074170642671101, 0.0007018754584836353, 0.0006948635827350929, 0.000687047051783197, 0.0006794219538525341, 0.0006732698993902021, 0.0006700179233731207, 0.000670973997351924, 0.0006769218572661918, 0.0006875886419744119, 0.0007010534513029816, 0.0007132502694798772, 0.0007178263454483512, 0.0007067015598390462, 0.0006716333238827775, 0.0006067928316921417, 0.0005117626882441227, 0.0003937066214574832, 0.0002672587467361692, 0.00015135403591180115, 6.356896748989911e-5, 1.3819222806010018e-5, 0.0])
        pd_n2v = np.array([1.0965111905839644e-5, 1.0121419262408943e-5, 9.204413404990034e-6, 8.206514788226718e-6, 7.11940487595017e-6, 5.933996707050697e-6, 4.64042053039306e-6, 3.2280312419306533e-6, 1.685446251422291e-6, 6.246971004417558e-10, -1.8389983415513366e-6, -3.846305779060872e-6, -6.034200071340727e-6, -8.415187714183227e-6, -1.100081756519295e-5, -1.3800939254260953e-5, -1.6822743545976724e-5, -2.006954287183131e-5, -2.3539248192177516e-5, -2.7222499084062128e-5, -3.110040917368999e-5, -3.51419010432006e-5, -3.930062657495703e-5, -4.3511504219242324e-5, -4.7686958616481405e-5, -5.171302587454759e-5, -5.5445595528837434e-5, -5.870720343384351e-5, -6.128497255244937e-5, -6.293052096803016e-5, -6.336291128798399e-5, -6.227598347673108e-5, -5.9351655508673324e-5, -5.4280926986391336e-5, -4.679427711328254e-5, -3.670276240102519e-5, -2.395019620015428e-5, -8.675099503139283e-6, 8.72158500920326e-6, 2.751089453028125e-5, 4.658526728366228e-5, 6.44416830030953e-5, 7.922281846969817e-5, 8.884413006063435e-5, 9.123368539824376e-5, 8.470217810632975e-5, 6.843886854594554e-5, 4.309176347052518e-5, 1.1335988095736056e-5, -2.173329774801049e-5, -4.861532014870233e-5, -6.009885359885891e-5, -4.68781370560342e-5, -2.1526193083744357e-6, 7.51123115183263e-5, 0.00017717545929127017, 0.00028627768540146777, 0.0003768668198682145, 0.00042219838125253056, 0.00040424856847082224, 0.00032336124205629155, 0.0002024371238746584, 8.121616638627767e-5, 0.0])

        pd_n2_inf = -6.12066824743757e-5
        pd_n3_inf = -1.555055491609012e-5
        pd_n2v_inf = -2.3438609691423285e-5

        pd_disp = np.array([-0.2817430904524846, -0.2817384633532139, -0.28173284708951124, -0.28172603796458706, -0.2817177920713764, -0.2817078178601837, -0.2816957674979746, -0.2816812268808188, -0.28166370416755054, -0.2816426167210397, -0.28161727637854933, -0.2815868730313236, -0.28155045658440186, -0.2815069175018294, -0.2814549663336322, -0.2813931128859961, -0.2813196460550091, -0.28123261582000725, -0.2811298195100848, -0.2810087952422698, -0.2808668264050108, -0.2807009622409254, -0.2805080609668174, -0.280284863426884, -0.28002810693026686, -0.27973469052901695, -0.27940190429501627, -0.27902773575634005, -0.2786112659604752, -0.27815316479807056, -0.27765628909807133, -0.2771263761090879, -0.2765728075055793, -0.2760093929703852, -0.27545508570071925, -0.27493449335602765, -0.27447798680600216, -0.27412113785801445, -0.27390314247018743, -0.27386382063891807, -0.2740387496136791, -0.27445211507411776, -0.2751069974522217, -0.2759730964114608, -0.27697238115806605, -0.27796386143140656, -0.2787295763142595, -0.2789648792201242, -0.27827691519453956, -0.2761954557567532, -0.27219948659594934, -0.26576067537337217, -0.25640092284642463, -0.24375615453596464, -0.22763395083319907, -0.2080512894833483, -0.18524373320435525, -0.15965036726897638, -0.1318964031202509, -0.10280672023157833, -0.0734713901128339, -0.04533456256504357, -0.02020065722389818, 0.0])
        pd_disp_inf = -0.021246311698371732


        pd_n2_out = np.array([0.16089139212716583, 0.15960614912104462, 0.15819821964065686, 0.15665680034071, 0.1549703324825682, 0.15312649362344222, 0.15111220150396767, 0.14891363364570767, 0.14651626684922234, 0.14390494155029904, 0.14106395684509476, 0.13797720292635693, 0.13462833866546367, 0.1310010230988328, 0.1270792105848068, 0.12284752031749689, 0.11829169161559391, 0.113399136805831, 0.10815960340232159, 0.10256595639393473, 0.09661508947046707, 0.09030897054164236, 0.08365582144009374, 0.07667142367206652, 0.06938053083394732, 0.06181835315149119, 0.054032059841466924, 0.0460822200818497, 0.038044073026090745, 0.030008481743758714, 0.022082386289811885, 0.01438852964500195, 0.007064191180145686, 0.00025863213499791656, -0.0058710540588889455, -0.011165973888371887, -0.015473567512447274, -0.018657927623459674, -0.020612163013917193, -0.021272336984547474, -0.020632067692572535, -0.0187563321726597, -0.01579241141639848, -0.011975340837472278, -0.007624839265544469, -0.0031306967261977747, 0.0010757237618247086, 0.00456459436954766, 0.006963303071436723, 0.008018800825121478, 0.007656186957402335, 0.006018983546487027, 0.0034745651433051267, 0.0005705791233883081, -0.002063042922263479, -0.003856073651896031, -0.004453079718759342, -0.0038322424493157166, -0.002332848090384517, -0.000554838401791689, 0.0008590400013082883, 0.0014836799545610087, 0.0012908761806870817, 0.0006336071240813062])

        pd_n3_out = np.array([0.0445972264060453, 0.04424196728743102, 0.04385279255688607, 0.04342671498448464, 0.04296053829390522, 0.04245085481309086, 0.04189404648649914, 0.0412862902177814, 0.04062356869919306, 0.039901688095770126, 0.03911630418786667, 0.038262958832856304, 0.03733712888095355, 0.036334289962973707, 0.035249997846348316, 0.034079990310308575, 0.03282031269383566, 0.03146747038190769, 0.03001861146431427, 0.02847174255784701, 0.02682598023834676, 0.025081839572842823, 0.023241559738599545, 0.021309464505894778, 0.019292352264653718, 0.01719990609945487, 0.01504510897541204, 0.012844642234491862, 0.010619237240512815, 0.008393940215716421, 0.0061982393783744085, 0.004065992065250108, 0.0020350787467496773, 0.00014670252058790082, -0.0015557505648249158, -0.0030283695630532836, -0.004228955574567622, -0.005119860803190618, -0.0056713914255312175, -0.005865645442153174, -0.005700534954332144, -0.0051935925030006215, -0.004384995147682909, -0.003339082660327824, -0.0021435389015160853, -0.0009054074825294303, 0.00025670198567362087, 0.0012244164457630741, 0.001894853718008229, 0.0021977129757968948, 0.0021113496384006446, 0.00167385874377928, 0.0009846527256066487, 0.0001926714104360063, -0.0005302426847519516, -0.001027962844753716, -0.0012022822489440315, -0.0010452987932170284, -0.000647071490273874, -0.00016868600308945898, 0.00021635561619680544, 0.0003921989515710299, 0.0003488911488462498, 0.00017847282202198048])

        pd_n2v_out = np.array( [0.005496057154802355, 0.005723576478445079, 0.005955606150424454, 0.006191388748638015, 0.00642999684899994, 0.006670309378125141, 0.006910985813304674, 0.00715043831163237, 0.007386801961338955, 0.007617903497079471, 0.007841229015916704, 0.008053891482143943, 0.008252599128147659, 0.008433626257313838, 0.008592788445839826, 0.008725424734716289, 0.008826390110118543, 0.008890062394102163, 0.008910368603473872, 0.008880836865225436, 0.008794681064068114, 0.008644926474559321, 0.008424585590172791, 0.00812689404389917, 0.0077456166897826185, 0.007275433266469304, 0.0067124111751440604, 0.006054569245904574, 0.005302530298185194, 0.00446025109246674, 0.003535805159913836, 0.0025421762997266083, 0.0014979978329849408, 0.0004281451361077231, -0.0006359423379038694, -0.0016563657582088958, -0.00258948320087244, -0.003387439495170931, -0.004000705849538491, -0.004381781852094353, -0.004490113065906577, -0.004298103992897659, -0.0037978448973957973, -0.0030078210140509925, -0.0019784581936152295, -0.0007949442264270644, 0.0004245314355050017, 0.0015369750732279093, 0.0023905500916872346, 0.002848345240771134, 0.0028178253156006276, 0.002280920901451645, 0.0013166240292247044, 0.00010587277608634535, -0.0010908498070021655, -0.0019875972525751355, -0.002352515842492401, -0.002092194183443678, -0.0013071159213224734, -0.0002820044454515453, 0.0006078110288443224, 0.0010532998822645643, 0.0009628755279201839, 0.0005137813102362146] )

        pd_disp_out = np.array([-16.83391424645579, -16.701183314944448, -16.555810079798665, -16.396687199414227, -16.222631201663322, -16.032381989899477, -15.824603667654953, -15.597887050698077, -15.350754304502285, -15.08166622289612, -14.789032749243233, -14.47122743372796, -14.126606616925738, -13.753534227091473, -13.350413170815154, -12.915724375599178, -12.448074596898406, -11.946254115419151, -11.40930540201209, -10.836603690149468, -10.22795013545463, -9.58367781593076, -8.904770185289815, -8.192990677749691, -7.451020913565925, -6.68260330787128, -5.892681786469393, -5.087531727170943, -4.274867180034295, -3.463909947140961, -2.6654013973596666, -1.8915342760378206, -1.155778764509531, -0.47257542232590666, 0.1431315260931788, 0.6765424422709011, 1.1138438194412008, 1.44323753280696, 1.656126067076576, 1.7483862162992154, 1.721618980611451, 1.5842128269383908, 1.352008130651045, 1.048314399451782, 0.7030261893763766, 0.350630383240835, 0.027019651894398974, -0.23475893914972276, -0.40938668594574695, -0.4839943014141165, -0.461597885185548, -0.3619648441245001, -0.21881023268433125, -0.07291862395186478, 0.03796546787765717, 0.09004272128389004, 0.08135340312395764, 0.03259835034743712, -0.02141878430328844, -0.04801184928438945, -0.033693940404021946, 0.007843303080272625, 0.04433101273697712, 0.047904865515183026] )
        pd_b_out = -0.023417547028948233

        pd_overall = np.array([-0.1945995531718032, -0.19455448633435599, -0.1945050881522831, -0.19445097107081405, -0.1943917190963474, -0.19432688713079177, -0.19425600067301702, -0.19417855600357656, -0.19409402099269218, -0.1940018366986406, -0.19390141995413582, -0.19379216717174047, -0.19367345963504007, -0.19354467057892052, -0.1934051743975368, -0.1932543583488476, -0.19309163714453564, -0.19291647081575033, -0.19272838621697813, -0.19252700245636498, -0.19231206039866808, -0.1920834561463314, -0.191841278024248, -0.1915858460209865, -0.19131775180445915, -0.19103789624596731, -0.19074751974647286, -0.19044821843700832, -0.190141936381156, -0.1898309200986067, -0.18951761693249353, -0.18920449293836708, -0.18889373914646262, -0.1885868275399092, -0.1882838705804263, -0.1879827318701221, -0.18767783268948912, -0.18735860298990395, -0.18700754070704206, -0.18659787642558998, -0.18609089943520657, -0.18543309483159767, -0.18455337721651494, -0.18336088775581097, -0.1817440402941807, -0.1795717321319794, -0.17669781925504247, -0.17296999725276452, -0.16824398606900984, -0.16240321422409257, -0.15538287093512435, -0.1471951771835342, -0.13795020032186844, -0.12786411461449107, -0.11724569018865087, -0.10645372246842605, -0.09582493120927936, -0.08558437920112245, -0.0757667193321722, -0.0661900404333151, -0.056523441271515644, -0.04646244195254592, -0.0359692355803277, -0.025465485850310634])

        return pd_n2, pd_n2_inf, pd_n3, pd_n3_inf, pd_n2v, pd_n2v_inf, pd_disp, pd_disp_inf, pd_n2_out, pd_n3_out, pd_n2v_out, pd_disp_out, pd_b_out, pd_overall

    def test_weigthed_densities(self, do_feos=False):
        """
        Code testing

        """
        #self.setup_transforms()
        #self.calc_lanczos()
        #self.analytical_fourier_weigths()
#and self.z_edge[i] > 0.0
        rho = np.zeros(self.n_grid)
        for i in range(self.n_grid):
            if self.z_edge[i] < 1.5:
                rho[i] = 3.0/np.pi
        if do_feos:
            rho, n2_feos, n3_feos, n2v_feos, n_disp_feos = self.get_feos()
        #rho = np.ones(self.n_grid)
        # np.ones(np.shape(rho)[0])*
        rho_inf = rho[-1]
        #print("rho_inf",rho_inf)
        rho_delta = np.zeros_like(rho)
        rho_delta[:] = rho - rho_inf
        frho_delta = self.transform(rho_delta, scalar=True, forward=True)
        f2 = np.zeros_like(rho)
        f2[:] = frho_delta*self.fw2
        n2 = self.transform(f2, scalar=True, forward=False)
        n2_inf = rho_inf*self.fw2_inf
        n2 += n2_inf

        f3 = np.zeros_like(rho)
        f3[:] = frho_delta*self.fw3
        n3 = self.transform(f3, scalar=True, forward=False)
        n3_inf = rho_inf*self.fw3_inf
        n3 += n3_inf

        f2v = np.zeros_like(rho)
        f2v[:] = frho_delta*self.fw2vec
        n2v = -self.transform(f2v, scalar=False, forward=False) #...... sign

        f_disp = np.zeros_like(rho)
        f_disp[:] = frho_delta*self.fw_disp
        n_disp = self.transform(f_disp, scalar=True, forward=False)
        n_disp_inf = rho_inf*self.fw_disp_inf
        n_disp += n_disp_inf

        plt.plot(self.z,rho,label="rho")
        plt.plot(self.z,n2/(4 * np.pi * self.R ** 2),label="n0")
        plt.plot(self.z,n3,label="n3")
        plt.plot(self.z,n2v,label="n2v")
        plt.plot(self.z,n_disp,label="n_disp")

        if do_feos:
            #plt.plot(self.z,n2_feos/(4 * np.pi * self.R ** 2),label="n0_feos")
            plt.plot(self.z,n3_feos,label="n3_feos")
            #plt.plot(self.z,n2v_feos,label="n2v_feos")
            #plt.plot(self.z,n_disp_feos,label="n_disp_feos")

        plt.legend(loc="best")
        plt.show()

    def test_pd(self):
        """
        Code testing

        """
        pd_n2, pd_n2_inf, pd_n3, pd_n3_inf, pd_n2v, pd_n2v_inf, pd_disp, pd_disp_inf, pd_n2_out, pd_n3_out, pd_n2v_out, pd_disp_out, pd_b_out, pd_overall = self.get_feos_pd()
        inf_contr = 0.0
        fd3 = self.transform(pd_n3, scalar=True, forward=True)
        # print("fd3",fd3)
        # print("pd_n3_out",pd_n3_out)
        f3 = np.zeros_like(pd_n3)
        f3 = fd3*self.fw3
        #diff_3 = self.transform(f, scalar=True, forward=False)
        inf_contr = pd_n3_inf*self.fw3_inf

        fd2 = self.transform(pd_n2, scalar=True, forward=True)
        # print("fd2",fd2)
        # print("pd_n2_out",pd_n2_out)
        f2 = np.zeros_like(pd_n3)
        f2 = fd2*self.fw2
        #diff_2 = self.transform(f, scalar=True, forward=False)
        inf_contr += pd_n2_inf*self.fw2_inf

        #print("pd_n2v",pd_n2v)
        fd2v = self.transform(pd_n2v, scalar=False, forward=True)
        # print("fd2v",fd2v)
        # print("pd_n2v_out",pd_n2v_out)
        f2v = np.zeros_like(pd_n3)
        f2v = fd2v*self.fw2vec
        #diff_2v = self.transform(f, scalar=True, forward=False)

        fd_disp = self.transform(pd_disp, scalar=True, forward=True)
        #print("fd_disp",fd_disp)
        #print("pd_disp_out",pd_disp_out)
        f_disp = np.zeros_like(pd_n3)
        f_disp = fd_disp*self.fw_disp
        #diff_disp = self.transform(f, scalar=True, forward=False)
        inf_contr += pd_disp_inf*self.fw_disp_inf

        f = f3 + f2 + f2v + f_disp
        diff_disp = self.transform(f, scalar=True, forward=False)
        print("diff_disp",diff_disp)
        print("pd_overall",pd_overall)

        # #print("inf_contr",inf_contr)

    def setup_transforms(self):
        """
        Code testing

        """
        n_grid = self.n_grid
        x0 = self.x0
        alpha = self.alpha
        self.gamma = np.exp(alpha * (n_grid - 1))
        gamma = self.gamma
        l = self.b
        self.k_grid = np.zeros(n_grid)
        for i in range(n_grid):
            self.k_grid[i] = x0 * np.exp(alpha * i) * gamma / l


        #print("k_grid", k_grid)
        self.j = np.zeros(2*n_grid, dtype=np.cdouble)
        self.jv = np.zeros(2*n_grid, dtype=np.cdouble)
        for i in range(2*n_grid):
            self.j[i] = jv(1, gamma * x0 * np.exp(alpha * (i + 1 - n_grid))) # / (2 * n_grid)
            self.jv[i] = jv(2, gamma * x0 * np.exp(alpha * (i + 1 - n_grid))) # / (2 * n_grid)

        #print("j", self.j)
        #print("jv", self.jv)
        self.j = ifft(self.j)
        self.jv = ifft(self.jv)
        #print(self.jv)

    def analytical_fourier_weigths(self):
        """

        """
        N = self.n_grid
        # Fourier space variables
        kz_abs = np.zeros_like(self.k_grid)
        kz_abs[:] = self.R*self.k_grid
        # self.fw3 = np.zeros(N, dtype=np.cdouble)
        # self.fw2 = np.zeros(N, dtype=np.cdouble)
        # self.fw2vec = np.zeros(N, dtype=np.cdouble)
        # self.fw_disp = np.zeros(N, dtype=np.cdouble)

        self.fw3 = np.zeros(N)
        self.fw2 = np.zeros(N)
        self.fw2vec = np.zeros(N)
        self.fw_disp = np.zeros(N)

        # self.fw3.real[:] = (4.0/3.0) * np.pi * self.R**3 * \
        #     (spherical_jn(0, kz_abs) + spherical_jn(2, kz_abs))
        # self.fw3.imag[:] = 0.0
        # self.fw2.real[:] = 4 * np.pi * self.R**2 * spherical_jn(0, kz_abs)
        # self.fw2.imag[:] = 0.0
        # self.fw2vec.real[:] = 0.0
        # self.fw2vec.imag[:] = -self.k_grid * self.fw3.real[:]

        self.fw3[:] = (4.0/3.0) * np.pi * self.R**3 * \
            (spherical_jn(0, kz_abs) + spherical_jn(2, kz_abs))
        self.fw2[:] = 4 * np.pi * self.R**2 * spherical_jn(0, kz_abs)
        self.fw2vec[:] = -self.k_grid * self.fw3[:]
        #
        psi_disp=1.3862
        kz_abs *= psi_disp*2
        self.fw_disp[:] = (spherical_jn(0, kz_abs) + spherical_jn(2, kz_abs))
        #self.fw_disp.imag = 0.0

        kz_abs0 = np.zeros(1)
        self.fw2_inf = 4 * np.pi * self.R**2 * spherical_jn(0, kz_abs0)[0]
        self.fw3_inf = (4.0/3.0) * np.pi * self.R**3 * \
            (spherical_jn(0, kz_abs0) + spherical_jn(2, kz_abs0))[0]
        self.fw2vec_inf = 0.0
        self.fw_disp_inf = 1.0 #(spherical_jn(0, kz_abs0) + spherical_jn(2, kz_abs0))

        # print("self.fw2_inf",self.fw2_inf)
        # print("self.fw3_inf",self.fw3_inf)
        # print("self.fw2vec_inf",self.fw2vec_inf)
        # print("self.fw_disp_inf",self.fw_disp_inf)

        self.fw2 *= self.lanczos_sigma
        self.fw3 *= self.lanczos_sigma
        self.fw2vec *= self.lanczos_sigma
        self.fw_disp *= self.lanczos_sigma
        # print("self.fw2.real",self.fw2)
        #print("self.fw3.real",self.fw3)
        # print("self.fw2vec.imag",self.fw2vec)
        # print("self.fw_disp.real",self.fw_disp)

    def calc_lanczos(self):
        n = self.n_grid
        m2 = float(n)
        m2 += 2.0 if n % 2 == 0 else 1.0
        self.lanczos_sigma = spherical_jn(0, self.k_grid * self.b / m2)
        #self.lanczos_sigma = np.ones(n)
        #print("self.lanczos_sigma",self.lanczos_sigma)

    def transform(self, f_in, scalar=True, forward=True):
        """
        Code testing

        """
        #print("f_in",f_in)
        N = self.n_grid
        f = np.zeros_like(f_in)
        alpha = self.alpha
        if forward:
            factor = self.b
            x_in = self.z
            x_out = self.k_grid
        else:
            factor = self.gamma / self.b
            x_in = self.k_grid
            x_out = self.z

        if scalar:
            f[:] = f_in[:]
            k0 = self.k0
            j = self.j
        else:
            alpha *= 2.0
            k0 = self.kv0
            j = self.jv
            factor = factor**2
            f[:] = f_in[:] / x_in

        #k0 = 1.0
        # print("factor",factor)
        # print("alpha",alpha)
        # print("k0",k0)
        phi = np.zeros(2*N, dtype=np.cdouble)
        for i in range(N-1):
            phi[i] = (f[i] - f[i + 1]) * np.exp(-alpha * (N - i - 1))
        phi[0] *= k0

        #print("phi",phi)
        fphi = fft(phi)
        #print("fphi",fphi)
        fphi *= j
        #print("phij",fft(fphi)) # fft(fphi).real[:N]
        f_out = fft(fphi).real[:N] * factor / x_out
        #print("f_out",f_out)
        return f_out

class polar_weights(Weights):
    """
    """

    def __init__(self, dr: float, R: float, N: int):
        """

        Args:
            dr (float): Grid spacing
            R (float): Particle radius
            N (int): Grid size
        """
        Weights.__init__(self, dr, R, N)
        self.pol = polar(R, domain_size=self.L,n_grid=N)
        self.pol.setup_transforms()
        self.pol.calc_lanczos()
        self.pol.analytical_fourier_weigths()

        self.rho_delta = np.zeros(N)
        self.f = np.zeros(N)
        self.d3_delta = np.zeros(N)
        self.d2eff_delta = np.zeros(N)
        self.d2veff_delta = np.zeros(N)

    def convolutions(self, densities: weighted_densities_1D, rho: np.ndarray):
        """
        We here calculate the convolutions for the weighted densities for the
        spherical geometry

        Args:
            densities:
            rho (np.ndarray): Density profile

        """

        # Split into two terms such that rho_delta=0 when z-> inf.
        rho_inf=rho[-1]
        self.rho_delta=rho-rho_inf

        # Fourier transfor only the rho_delta
        frho_delta = self.pol.transform(self.rho_delta, scalar=True, forward=True)

        # 2d weighted density
        self.f = frho_delta*self.pol.fw2
        densities.n2[:] = self.pol.transform(self.f, scalar=True, forward=False)
        n2_inf = rho_inf*self.pol.fw2_inf
        densities.n2[:] += n2_inf

        # 3d weighted density
        self.f = frho_delta*self.pol.fw3
        densities.n3[:] = self.pol.transform(self.f, scalar=True, forward=False)
        n3_inf = rho_inf*self.pol.fw3_inf
        densities.n3[:] += n3_inf

        # 2d vector weighted density
        self.f = frho_delta*self.pol.fw2vec
        densities.n2v[:] = self.pol.transform(self.f, scalar=False, forward=False)
        #n2v_inf = rho_inf*self.pol.fw2vec_inf
        #densities.n2v[:] += n2v_inf

        # Calculate remainig weighted densities after convolutions
        densities.update_after_convolution()

    def convolution_n3(self, densities: weighted_densities_1D, rho: np.ndarray):
        """

            Args:
                densities:
                rho (np.ndarray): Density profile

            Returns:

            """

        # Split into two terms such that rho_delta=0 when z-> inf.
        rho_inf=rho[-1]
        self.rho_delta=rho-rho_inf

        # Fourier transfor only the rho_delta
        frho_delta = self.pol.transform(self.rho_delta, scalar=True, forward=True)

        # 3d weighted density
        self.f = frho_delta*self.pol.fw3
        densities.n3[:] = self.pol.transform(self.f, scalar=True, forward=False)
        n3_inf = rho_inf*self.pol.fw3_inf
        densities.n3[:] += n3_inf


    def correlation_convolution(self, diff: differentials_1D):
        """

        Args:
            diff: Functional differentials

        Returns:

        """

        # Split all terms into (a_delta+a_inf) such that a_delta=0 when z-> inf.
        d3_inf = diff.d3[-1]
        d2eff_inf = diff.d2eff[-1]
        d2veff_inf = diff.d2veff[-1]

        self.d3_delta=diff.d3-d3_inf
        self.d2eff_delta=diff.d2eff-d2eff_inf
        self.d2veff_delta=diff.d2veff-d2veff_inf

        # Fourier transfor only the d3_delta
        fd3 = self.pol.transform(self.d3_delta, scalar=True, forward=True)
        self.f = fd3*self.pol.fw3
        diff.d3_conv[:] = self.pol.transform(self.f, scalar=True, forward=False)
        d3_inf = d3_inf*self.pol.fw3_inf
        diff.d3_conv[:] += d3_inf

        # Fourier transfor only the d2eff_delta
        fd2 = self.pol.transform(self.d2eff_delta, scalar=True, forward=True)
        self.f = fd2*self.pol.fw2
        diff.d2eff_conv[:] = self.pol.transform(self.f, scalar=True, forward=False)
        d2eff_inf = d2eff_inf*self.pol.fw2_inf
        diff.d2eff_conv[:] += d2eff_inf

        # Fourier transfor only the d2veff_delta
        fd2v = self.pol.transform(self.d2veff_delta, scalar=False, forward=True)
        self.f = -fd2v*self.pol.fw2vec
        diff.d2veff_conv[:] = self.pol.transform(self.f, scalar=True, forward=False)
        #d2eff_inf = d2veff_inf*self.pol.fw2vec_inf
        #diff.d2veff_conv[:] += d2veff_inf

        diff.update_after_convolution()

    def fourier_weigths(self):
        """
        Fourier transform of w_2, w_3 and w_V2

        """
        n_grid = self.n_grid
        x0 = self.pol.x0
        alpha = self.pol.alpha
        gamma = np.exp(alpha * (n_grid - 1))
        l = self.b
        k_grid = np.zeros(n_grid)
        for i in range(n_grid):
            k_grid[i] = x0 * np.exp(alpha * i) * gamma / l

        self.j = np.zeros(2*n_grid, dtype=np.cdouble)
        self.jv = np.zeros(2*n_grid, dtype=np.cdouble)
        for i in range(2*n_grid):
            self.j[i] = jv(1, gamma * x0 * np.exp(alpha * (i + 1 - n_grid))) # / (2 * n_grid)
            self.jv[i] = jv(2, gamma * x0 * np.exp(alpha * (i + 1 - n_grid))) # / (2 * n_grid)

        self.j = ifft(self.j)
        self.jv = ifft(self.jv)


class polar_pc_saft_weights(polar_weights):
    """
    """

    def __init__(self, dr: float, R: float, N: int, phi_disp=1.3862):
        """

        Args:
            dr (float): Grid spacing
            R (float): Particle radius
            N (int): Grid size
            phi_disp (float): Weigthing distance for disperesion term
        """
        # Fourier space variables
        self.fw_disp = np.zeros(N)
        self.frho_disp_delta = np.zeros(N)
        self.fw_rho_disp_delta = np.zeros(N)
        self.fmu_disp_delta = np.zeros(N)
        self.fw_mu_disp_delta = np.zeros(N)

        #  Regular arrays
        self.mu_disp_inf=np.zeros(N)
        self.mu_disp_delta=np.zeros(N)

        # Weigthing distance for disperesion term
        self.phi_disp = phi_disp

        polar_weights.__init__(self, dr, R, N)

    def analytical_fourier_weigths(self):
        """
        Analytical Fourier transform of w_2, w_3 and w_V2
        For the 1D spherical transform

        """
        return

    def convolutions(self, densities: weighted_densities_pc_saft_1D, rho: np.ndarray):
        """

        Args:
            densities:
            rho (np.ndarray): Density profile

        """
        polar_weights.convolutions(self, densities, rho)

        # Split into two terms such that rho_delta=0 when z-> inf.
        rho_inf=rho[-1]
        self.rho_delta=rho-rho_inf

        # Fourier transfor only the rho_delta
        frho_delta = self.pol.transform(self.rho_delta, scalar=True, forward=True)

        # Dispersion weight
        self.f = frho_delta*self.pol.fw_disp
        densities.rho_disp[:] = self.pol.transform(self.f, scalar=True, forward=False)
        n_disp_inf = rho_inf*self.pol.fw_disp_inf
        densities.rho_disp[:] += n_disp_inf


    def correlation_convolution(self, diff: differentials_pc_saft_1D):
        """

        Args:
            diff: Functional differentials

        Returns:

        """
        polar_weights.correlation_convolution(self, diff)

        # Split the term into (a_delta+a_inf) such that a_delta=0 when z-> inf.
        mu_disp_inf = diff.mu_disp[-1]
        self.mu_disp_delta=diff.mu_disp - mu_disp_inf

        # Fourier transfor only the mu_delta
        fmu_disp_delta = self.pol.transform(self.mu_disp_delta, scalar=True, forward=True)

        # Dispersion weight
        self.f = fmu_disp_delta*self.pol.fw_disp
        diff.mu_disp_conv[:] = self.pol.transform(self.f, scalar=True, forward=False)
        mu_disp_inf_conv = mu_disp_inf*self.pol.fw_disp_inf
        diff.mu_disp_conv[:] += mu_disp_inf_conv

        diff.update_after_convolution()

if __name__ == "__main__":
    pass

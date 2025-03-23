
np.set_printoptions(threshold=np.inf)

structure = Atoms('340Au',positions)

# Setting up the Soap descriptor
soap = SOAP (
    species =[79],
    periodic = False,
rcut = 6.55,
nmax = 17,
lmax = 9,
rbf  = "gto"
)

# Calculate

soap_structure = soap.create(structure)
#print(soap_structure)
soap_structure = np.array(soap_structure)
filename = sys.argv[1]+"_soap_result.npy"
print(filename,soap_structure.shape)
np.save(filename,soap_structure) 

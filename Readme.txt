Doc. BA Repository

**HAVOK**

'ecg_gen' erstellt das HAVOK Modell mit Input einer EKG Zeitreihe
'ecg_analyse' analysiert das HAVOK Modell hinsichtlich der in der BA erarbeiteten Aspekte

**NN**

SINDy Autoencoder auf code basis von https://github.com/kpchamp/SindyAutoencoders

Konstruiert ein Autoencoder Netzwerk, das eine EKG-Zeitreihe Delay einbettet und mittels Autoencoder in
intrinsche Koordinaten einbettet und modelliert.

Als Input wird eine EKG-Zeitreihe gegeben, diese wird mittels get_hankel() in eine Hankelmatrix geschrieben
Die Parameter für das Training werden in das Notebook params[] geschrieben
in der analys_ecg.py werden die Losses und das korrespondierende Modell ausgegeben.

**Synthesized_governing_equasions**

Skript, das mittels der PySINDy Methode einen gegebenen Satz an Zuständen auf Systemgleichungen untersucht.
Inputs:
get_data_from_matlab 		#synthetisierte Daten
enable_jitter			#HRV on/off
new_data 			#Echtdaten
analyse_modes			#U-Moden der SVD untersuchen
add_constraints 		#Physical Constraints mit reinbringen
analyse_lamda 			#ParetoKurven für Datensatz erstellen
savedata_mat 			#trajektorien als mat speichern
compare_differentiation_method  #unterschied der differenzierungsmethode

Das errechnete Systeme wird geplottet und in der Konsole ausgegeben. 
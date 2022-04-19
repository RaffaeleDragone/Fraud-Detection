# FraudDetection
# Introduzione
Miliardi di euro vengono persi ogni anno a causa di
transazioni fraudolente con carta di credito

È importante che le banche siano in grado di riconoscere
questo tipo di transazioni in modo da non addebitare ai
propri clienti i costi di articoli che non hanno acquistato

Lo scopo di di FraudDetection è quello di costruire un
classificatore in grado di rilevare transazioni fraudolente
con carta di credito.

Le banche producono quotidianamente grandi quantità di
dati relativi alle transazioni che interessano i loro circuiti

I dataset finanziari sono importanti per i ricercatori per
effettuare ricerche nel campo del rilevamento delle frodi

Il dataset contiene una simulazione di transazioni
effettuate attraverso sistemi di pagamento mobile basata
su transazioni reali

# Dataset: 
## FinancialDatasetsForFraudDetection1

Il dataset presenta 6.362.621 transazioni totali avvenute
in due giorni di cui una parte è stata identificata come
fraudolente

## Dataset : Campi
step ( integer ): Unita di tempo del trasferimento;

type string categorical Tipologia di trasferimento;

amount (float): Importo del trasferimento;

nameOrig(string) : Informazioni sul mittente;

oldbalanceOrg (float) :  Bilancio iniziale del mittente;

newbalanceOrig (float): Bilancio dopo il trasferimento del
mittente;

nameDest (string) Informazioni sul destinatario;

oldbalanceDest (float): Bilancio iniziale del destinatario;

newbalanceDest (float):  Bilancio dopo il trasferimento
del destinatario;

isFraud (boolean/ sono le transazioni effettuate
dagli agenti fraudolenti In questo specifico dataset il
comportamento fraudolento degli agenti mira a trarre
profitto prendendo il controllo, con l'obiettivo di svuotare
i fondi trasferendoli su un altro account

isFlaggedFraud (boolean/ contrassegna i tentativi
illegali, cioè i tentativi di trasferire un importo maggiore
di 200 000 in una singola transazione

# Obiettivi Del Progetto
- Preparazione dei dati

  - Normalizzazione

  - Features Selection

- Selezione del modello da utilizzare

- Addestramento del modello Random Forest / Alberi Decisionali.

- Valutazione del modello

- Parameter Tuning

- Predizione

- Analisi dei risultati

  - Rappresentazione dei risultati

  - Discussione dei risultati

  - Conclusioni e sviluppi futuri

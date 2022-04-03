import pandas as pd

# redenumim coloanele sub aceeasi conventie
def correct_data(data_frame):
    data_frame = data_frame.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', \
                        'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})
    return data_frame
    #print(data_frame.head())

# problema datelor este deosebirea descrierei de realitate
def types_of_fraud(data_frame):
    print(f"\nTipul tranzactiilor frauduloase: {list(data_frame.loc[data_frame.isFraud == 1].type.drop_duplicates().values)}") 
    # doar "CASH_OUT" & "TRANSFER"
    # Conform datasetului frauda este comisa cand mai intai transferand bani spre un account care apoi ii scoate

    df_fraud_transfer = data_frame.loc[(data_frame.isFraud == 1) & (data_frame.type == "TRANSFER")]
    df_fraud_cashout = data_frame.loc[(data_frame.isFraud == 1) & (data_frame.type == "CASH_OUT")]

    print (f"\nNumarul de TRANSFER-uri frauduloase = {len(df_fraud_transfer)}") # 4097
    print (f"\nNumarul de CASH_OUT-uri frauduloase = {len(df_fraud_cashout)}")  # 4116

    print("\n"+'#'*20 + " Fraudulent TRANSFER - CASH_OUT connection " + '#'*20)

    # din descrierea datei, comiterea fraudei implica mai intai efectuarea transferului la un cont (fraudulent) care apoi face CASH_OUT
    # CASH_OUT implica tranzactii cu un comerciant care plateste cash
    # prin proces din 2 etape, contul fraudulent vor fi ambele, destinatarul in TRANSFER si initiatorul in CASH_OUT. 
    print("\nIn cadrul tranzactiilor frauduloase, exista destinatii pentru TRANSFER-uri care sunt si initiatoare pentru CASH_OUT?  {}"
        .format((df_fraud_transfer.nameDest.isin(df_fraud_cashout.nameOrig)).any())) # False
    # rezulta ca modus-operandi a datei nu este exacta
    
    df_not_fraud = data_frame.loc[data_frame.isFraud == 0]

    # nici nameOrig, nici nameDest nu codifica contul comerciantilor in modul corespunzator
    print("\nTRANSFER-uri frauduloase ale caror conturi de destinatie sunt initiatoare de CASH_OUT-uri reale: \n\n{}"
        .format(df_fraud_transfer.loc[df_fraud_transfer.nameDest.isin(df_not_fraud.loc[df_not_fraud.type == "CASH_OUT"].nameOrig.drop_duplicates())]))
    
    print("\nTRANSFER fraudulent catre C423543548 a avut loc la step = 486, in timp ce CASH_OUT autentic din acest cont a avut loc mai devreme la step = {}"
    .format(df_not_fraud.loc[(df_not_fraud.type == "CASH_OUT") & (df_not_fraud.nameOrig == "C423543548")].step.values)) # 185
    # vom exclude aceste  pentru ca sunt inutile
    
    print('#'*83 + "\n")

# studiem originea isFlaggedFraud si anume ce determina ca el sa fie setat sau nu
def isFlaggedFraud_info(data_frame):
    print("\n"+'#'*20 + " isFlaggedFraud flag analisys " + '#'*20)

    print(f"\nTipul tranzactiilor in care isFlaggedFraud este setat: {list(data_frame.loc[data_frame.isFlaggedFraud == 1].type.drop_duplicates())}") 
    # apare doar la "TRANSFER"

    df_transfer = data_frame.loc[data_frame.type == "TRANSFER"]
    df_flagged = data_frame.loc[data_frame.isFlaggedFraud == 1]
    df_not_flagged = data_frame.loc[data_frame.isFlaggedFraud == 0]

    # Data este descrisa ca fiind setata cand se face o incercare de a TRANSFERa un "amount" mai mare de 200 000
    # In realitate, isFlaggedFraud poate sa ramana nesetat chiar daca se indeplineste aceasta conditie
    print(f"\nCantitatea minima tranzactionata cand isFlaggedFraud este setat = {df_flagged.amount.min()}") # 353874.22 
    print(f"\nCantitatea maxima tranzactionata in TRANSFER cand isFlaggedFraud nu este setat =\
    {df_transfer.loc[df_transfer.isFlaggedFraud == 0].amount.max()}") # 92445516.64

    # setarea acestul flag sa fie dependenta de faptul ca oldBalanceOrig = newBalnceOrig
    # newBalanceOrig face update doar daca tranzactia trece, dar isFlaggedFraud va fi setat inainte sa treaca tranzactia
    print(f"\nNumarul de TRANSFER-uri unde isFlaggedFraud = 0, dar si oldBalanceDest = newBalanceDest = 0:\
    {len(df_transfer.loc[(df_transfer.isFlaggedFraud == 0) & (df_transfer.oldBalanceDest == 0) & (df_transfer.newBalanceDest == 0)])}") # 4158


    print(f"\nMin, Max in oldBalanceOrig pentru isFlaggedFraud = 1 TRANSFER-uri: \
    {[round(df_flagged.oldBalanceOrig.min()), round(df_flagged.oldBalanceOrig.max())]}")

    print("\nMin, Max in oldBalanceOri pentru isFlaggedFraud = 0 TRANSFER-uri unde oldBalanceOrig = newBalanceOrig: {}"\
        .format([df_transfer.loc[(df_transfer.isFlaggedFraud == 0) & 
                (df_transfer.oldBalanceOrig == df_transfer.newBalanceOrig)].oldBalanceOrig.min(), \
                round(df_transfer.loc[(df_transfer.isFlaggedFraud == 0) & 
                    (df_transfer.oldBalanceOrig == df_transfer.newBalanceOrig)].oldBalanceOrig.max())]))
    
    # duplicatele numelor utilizatorilor nu exista acolo unde isFlaggedFraud este setat
    # dar exista in tranzactiile unde variabila nu este setata. 
    # deducem ca cei care au isFlaggedFraud setat au facut tranzactie o singura data.
    print("\nAu efectuat tranzactii de mai multe ori initiatorii tranzactiilor semnalate ca frauda? {}"\
        .format((df_flagged.nameOrig.isin(pd.concat([df_not_flagged.nameOrig, df_not_flagged.nameDest]))).any())) # False

    print("\nDestinatiile tranzactiilor marcate ca frauda au initiat alte tranzactii? {}"\
        .format((df_flagged.nameDest.isin(df_not_flagged.nameOrig)).any())) # False

    # deoarece doar 2 conturi destinatare din 16 care au variabila "isFlaggedFraud" setata
    # au fost conturi de destinatar mai mult de o data
    # intelegem ca setarea "isFlaggedFraud" este independenta de faptul daca
    # countul destinatar a mai fost folosit pana acum sau nu
    print("\nCate conturi de destinatie ale tranzactiilor marcate ca frauda au fost conturi de destinație de mai multe ori?: {}"\
    .format(sum(df_flagged.nameDest.isin(df_not_flagged.nameDest)))) # 2

    print('#'*70+"\n")
    # in urma analizei variabilei putem conclude ca "isFlaggedFraud" este setat intr-un mod aleator

# prezenta conturilor comerciantilor
def merchant_info(data_frame):
    print("\n"+'#'*20 + " Merchant analisys " + '#'*20)

    # se stie ca CASH_IN implica plata de comerciant (merchant) a carui nume are prefixul M
    # dar datasetul nu are comercianti care fac CASH_IN tranzactii catre utilizatori sau CASH_OUT
    print("\nExista comercianti printre conturile initiatoare pentru tranzactii CASH_IN? {}"
        .format((data_frame.loc[data_frame.type == "CASH_IN"].nameOrig.str.contains('M')).any())) # False

    print("\nExista comercianti printre conturile de destinatie pentru tranzactiile CASH_OUT? {}"
        .format((data_frame.loc[data_frame.type == "CASH_OUT"].nameDest.str.contains('M')).any())) # False

    # in realitate nu sunt comercianti intre conturi de initiatori
    # comerciantii sunt doar in account destinatari pentru toate PAYMENT-uri
    print(f"\nExista comercianti printre conturile de initiator? {data_frame.nameOrig.str.contains('M').any()}") # False

    print("\nExista tranzactii care au comercianti printre conturile de destinatie, altele decat tipul PAYMENT? {}"
        .format((data_frame.loc[data_frame.nameDest.str.contains('M')].type != "PAYMENT").any())) # False

    # intelegem ca intre conturile din nameOrig si nameDest, comerciantii cu prefixul 'M' apare in mod aleator
    print('#'*59+"\n")
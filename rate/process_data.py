import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

def pre_procesarlos(filename, distancia):
    data = []
    resumen = []
    with open(filename, "r") as f:
        empezo_resumen = False
        for line in f:
            line = line.strip()
            match = re.match(r'\[\s*ID\]', line)
            if(not match):
                match = re.match(r'- - - - - - - - - - - - - - - - - - - - - - - - -', line)
                if(match):
                    empezo_resumen = True

                if (not empezo_resumen):
                    match = re.match(r'\[\s*\d+\]\s+(\d*\.\d*)\-(\d*\.\d*)\s+sec\s+(\d*\.?\d*)\s*(Bytes|KBytes|MBytes)\s*(\d*\.?\d*)\s+(bits/sec|Kbits/sec|Mbits/sec)\s+(\d*)\s+(\d*\.?\d*)\s+(Bytes|KBytes|MBytes)', line)
                    # [  5]  23.00-24.00  sec   127 KBytes  1.04 Mbits/sec    1   22.6 KBytes      
                    if match:
                        interval_start = float(match.group(1))
                        interval_end = float(match.group(2))
                        transfer = float(match.group(3))
                        transfer_unit = match.group(4)          # Bytes|KBytes|MBytes
                        bitrate = float(match.group(5))
                        bitrate_unit = match.group(6)           # bits/sec|Kbits/sec|Mbits/sec
                        retrans = int(match.group(7))
                        cwnd = float(match.group(8))
                        cwnd_unit = match.group(9)              # Bytes|KBytes|MBytes


                        # Convertimos todo a Bytes y bits/sec si se desea
                        # iperf's Transfer column is in MebiBytes: 1,048,576's of 8-bit Bytes
                        transfer_bytes = transfer
                        if transfer_unit == "KBytes":
                            transfer_bytes = transfer * 1024        # Kibibyte to Byte
                        elif transfer_unit == "MBytes":
                            transfer_bytes = transfer * 1024 ** 2   # Mebibyte to Byte

                        # iperf's Bitrate column is in megabits: 1,000,000's of 1-bit bits
                        bitrate_bps = bitrate
                        if bitrate_unit == "Kbits/sec":
                            bitrate_bps = bitrate * 1e3
                        elif bitrate_unit == "Mbits/sec":
                            bitrate_bps = bitrate * 1e6

                        cwnd_bytes = cwnd
                        if cwnd_unit == "KBytes":
                            cwnd_bytes = cwnd * 1e3
                        elif cwnd_unit == "MBytes":
                            cwnd_bytes = cwnd * 1e6

                        data.append({
                            "distance": distancia,
                            "interval_start": interval_start,
                            "interval_end": interval_end,
                            "transfer": transfer_bytes,
                            "bitrate": bitrate_bps,
                            "retransmissions": retrans,
                            "cwnd": cwnd_bytes
                        })

                else:
                    match = re.match(r'\[\s*\d+\]\s+(\d*\.\d*)\-(\d*\.\d*)\s+sec\s+(\d*\.?\d*)\s*(Bytes|KBytes|MBytes)\s*(\d*\.?\d*)\s+(bits/sec|Kbits/sec|Mbits/sec)\s+(\d*)\s+(sender|receiver)', line)
                    # [  5]   0.00-60.00  sec  5.78 MBytes   807 Kbits/sec   37             sender
                    if match:
                        interval_start = float(match.group(1))
                        interval_end = float(match.group(2))
                        transfer = float(match.group(3))
                        transfer_unit = match.group(4)
                        bitrate = float(match.group(5))
                        bitrate_unit = match.group(6)
                        retrans = match.group(7)
                        type = match.group(8)

                        # Convertimos todo a Bytes y bits/sec si se desea
                        # iperf's Transfer column is in MebiBytes: 1,048,576's of 8-bit Bytes
                        transfer_bytes = transfer
                        if transfer_unit == "KBytes":
                            transfer_bytes = transfer * 1024            # Kibibyte to Byte
                        elif transfer_unit == "MBytes":
                            transfer_bytes = transfer * 1024 ** 2       # Mebibyte to Byte

                        # iperf's Bitrate column is in megabits: 1,000,000's of 1-bit bits
                        bitrate_bps = bitrate
                        if bitrate_unit == "Kbits/sec":
                            bitrate_bps = bitrate * 1e3
                        elif bitrate_unit == "Mbits/sec":
                            bitrate_bps = bitrate * 1e6

                        resumen.append({
                            "distance": distancia,
                            "interval_start": interval_start,
                            "interval_end": interval_end,
                            "transfer": transfer_bytes,
                            "bitrate": bitrate_bps,
                            "retransmissions": retrans,
                            "type": type
                        })

    return resumen, data

def post_procesarlos(df_data, df_resumen):
    datos_finales = []

    for distancia in DISTANCIAS:
        # print(f"------------------ {distancia} MTS ----------------------")

        total_time = df_resumen[(df_resumen["distance"]==distancia)]["interval_end"].iloc[0]
        total_active_time = (df_data[(df_data["bitrate"]>0) & (df_data["distance"]==distancia)]["interval_end"] - df_data[(df_data["bitrate"]>0) & (df_data["distance"]==distancia)]["interval_start"]).sum()
        # print(total_active_time)

        # print("-----TRANSFER-----")
        transf_suma_total = df_data[(df_data["distance"]==distancia)]["transfer"].sum()
        arr_transf = df_data[(df_data["distance"]==distancia) & (df_data["transfer"]>0)]["transfer"].to_numpy()
        transf_incert_muestral = np.std(arr_transf, ddof=1)

        transf_resumen_sender = df_resumen[(df_resumen["distance"]==distancia) & (df_resumen["type"]=="sender")]["transfer"].iloc[0]
        transf_resumen_receiver = df_resumen[(df_resumen["distance"]==distancia) & (df_resumen["type"]=="receiver")]["transfer"].iloc[0]
    
        # print(transf_suma_total)
        # print(transf_resumen_sender)
        # print(transf_resumen_receiver)

        # print("-----BITRATE-----")
        bitrate_prom_total = df_data[(df_data["distance"]==distancia)]["bitrate"].sum() / total_time
        bitrate_prom_val_pos = df_data[(df_data["bitrate"]>0) & (df_data["distance"]==distancia)]["bitrate"].sum() / total_active_time
        # si tenés un array de bitrates
        arr_bit = df_data[(df_data["bitrate"]>0) & (df_data["distance"]==distancia)]["bitrate"].to_numpy()
        # print(arr_bit)
        bitrate_incert_muestral = np.std(arr_bit, ddof=1)  # varianza muestral (n-1)
        # print(bitrate_incert_muestral)
        bitrate_resumen_sender = df_resumen[(df_resumen["distance"]==distancia) & (df_resumen["type"]=="sender")]["bitrate"].iloc[0]
        bitrate_resumen_receiver = df_resumen[(df_resumen["distance"]==distancia) & (df_resumen["type"]=="receiver")]["bitrate"].iloc[0]

        # print(bitrate_prom_total)
        # print(bitrate_prom_val_pos)
        # print(bitrate_resumen_sender)
        # print(bitrate_resumen_receiver)


        # print("-----DATOS PERDIDOS-----")
        cant_datos_totales=int(df_data[df_data["distance"]==distancia].shape[0])
        cant_datos_perdidos=int(df_data[(df_data["distance"]==distancia) & (df_data["bitrate"]==0)].shape[0])
        porcentaje_perdida_datos = 100*cant_datos_perdidos/cant_datos_totales
        # print(f"DATOS PERDIDOS:   {cant_datos_perdidos}/{cant_datos_totales} = {(porcentaje_perdida_datos):.2f}%")

        # print("------RETRANSMISIONES-------")
        cant_reenvios=int(df_resumen[(df_resumen["distance"]==distancia) & (df_resumen["type"]=="sender")]["retransmissions"].iloc[0])
        # print(f"CANTIDAD DE REENVIOS: {cant_reenvios}")

        datos_finales.append({
            "distancia": distancia, 
            "total_time": total_time, 
            "total_active_time": total_active_time,
            "porcentaje_perdida_datos": porcentaje_perdida_datos,
            "transf_suma_total": transf_suma_total, 
            "bitrate_prom_total": bitrate_prom_total, 
            "bitrate_prom_val_pos": bitrate_prom_val_pos, 
            "bitrate_incert_muestral": bitrate_incert_muestral
            })
        
    return datos_finales

def plot_bitrate_n_lostPackages(df):
    fig, ax1 = plt.subplots()

    # Eje para bitrates
    ax1.plot(df["distancia"], df["bitrate_prom_total"], "o-", label="Bitrate total")
    ax1.plot(df["distancia"], df["bitrate_prom_val_pos"], "s-", label="Bitrate valores >0")

    ax1.set_xlabel("Distancia (m)")
    ax1.set_ylabel("Bitrate (bits/s)")
    ax1.legend(loc="upper left")

    # Eje para pérdida en porcentaje
    ax2 = ax1.twinx()
    ax2.plot(df["distancia"], df["porcentaje_perdida_datos"], "r^-", label="Pérdida de datos (%)")
    ax2.set_ylabel("Pérdida de datos (%)", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    fig.tight_layout()
    plt.show()

def plot_bitrate(df):
    # crear la figura
    plt.figure(figsize=(7,4))
    plt.plot(df["distancia"], df["bitrate_prom_val_pos"], "o-", label="Bitrate valores >0")
    plt.fill_between(
        df["distancia"],
        df["bitrate_prom_val_pos"]+ df["bitrate_incert_muestral"],   # la incertidumbre (desvío estándar o similar)
        df["bitrate_prom_val_pos"]- df["bitrate_incert_muestral"],   # la incertidumbre (desvío estándar o similar)
        label="Bitrate con incertidumbre",
        alpha=0.2
    )
    plt.xlabel("Distancia (m)")
    plt.ylabel("Bitrate (bit/s)")
    plt.title("Bitrate vs. distancia")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    df[["distancia","bitrate_prom_val_pos","bitrate_incert_muestral"]].to_csv("registros/bitrate_vs_distancia.csv", index=False)


DISTANCIAS = [3, 6, 9, 12, 15, 18]

if __name__ == "__main__":

    resumen_total = []
    datos_totales = []

    for distancia in DISTANCIAS:
        archivo = "registros/rate_5dbm_"+str(distancia)+"mts"
        resumen_por_distancia, data_por_distancia = pre_procesarlos(filename=archivo, distancia=distancia)
        resumen_total.extend(resumen_por_distancia)
        datos_totales.extend(data_por_distancia)

    df_data = pd.DataFrame(datos_totales)
    df_resumen = pd.DataFrame(resumen_total)

    #df_data.to_csv("registros/data.csv", index=False)
    #df_resumen.to_csv("registros/resumen.csv", index=False)

    datos = post_procesarlos(df_data=df_data, df_resumen=df_resumen)

    # convertir lista de dict a DataFrame
    df = pd.DataFrame(datos)
    # ordenar por distancia para que el gráfico sea más prolijo
    df = df.sort_values("distancia")

    plot_bitrate_n_lostPackages(df=df)

    plot_bitrate(df=df)
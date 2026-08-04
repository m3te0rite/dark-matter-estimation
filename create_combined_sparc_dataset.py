import os
import pandas as pd
import numpy as np

# Base directory relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def build_combined_sparc_dataset(data_dir="data/train", output_csv="data/sparc_combined_train.csv"):
    """
    Combines the 4 separate SPARC observational datasets (galaxies.csv, stellar-masses.csv, 
    BTF.csv, and newt-mass.csv) into a unified master dataset for machine learning models.
    Preserves all 4 original CSV files in data_dir.
    """
    abs_data_dir = os.path.isabs(data_dir) and data_dir or os.path.join(BASE_DIR, data_dir)
    abs_output_csv = os.path.isabs(output_csv) and output_csv or os.path.join(BASE_DIR, output_csv)

    # Check if files are directly inside data_dir or inside data_dir/separate
    if not os.path.exists(os.path.join(abs_data_dir, "galaxies.csv")) and os.path.exists(os.path.join(abs_data_dir, "separate", "galaxies.csv")):
        abs_data_dir = os.path.join(abs_data_dir, "separate")

    print(f"Loading separate SPARC CSVs from '{abs_data_dir}'...")
    df_galaxies = pd.read_csv(os.path.join(abs_data_dir, "galaxies.csv"))
    df_stellar  = pd.read_csv(os.path.join(abs_data_dir, "stellar-masses.csv"))
    df_btf      = pd.read_csv(os.path.join(abs_data_dir, "BTF.csv"))
    df_newt     = pd.read_csv(os.path.join(abs_data_dir, "newt-mass.csv"))

    # Standardize galaxy identifiers across all files
    df_galaxies["Galaxy_clean"] = df_galaxies["Galaxy"].astype(str).str.strip().str.upper()
    df_stellar["Galaxy_clean"]  = df_stellar["Galaxy"].astype(str).str.strip().str.upper()
    df_btf["Galaxy_clean"]      = df_btf["Name"].astype(str).str.strip().str.upper()
    df_newt["Galaxy_clean"]     = df_newt["ID"].astype(str).str.strip().str.upper()

    # Aggregate point-wise rotation curve measurements per galaxy from newt-mass.csv
    def calc_newt_summary(group):
        group = group.sort_values("R")
        outer_row = group.iloc[-1]
        
        R_max   = outer_row["R"]
        V_obs   = outer_row["Vobs"]
        V_gas   = outer_row["Vgas"]
        V_disk  = outer_row["Vdisk"]
        V_bul   = outer_row["Vbul"]

        # Calculate Baryonic rotation velocity V_baryon^2 = V_gas^2 + V_disk^2 + V_bul^2
        V_baryon_sq = max(0.0, V_gas**2 + V_disk**2 + V_bul**2)
        V_obs_sq    = max(V_obs**2, 1e-5)

        # Dark Matter Fraction f_DM at outer radius
        dm_fraction = np.clip(1.0 - (V_baryon_sq / V_obs_sq), 0.0, 1.0)
        
        # Dynamical Mass M_dyn = (V_obs^2 * R_max) / G  [in Solar Masses]
        M_dyn_Msun  = 2.325e5 * V_obs_sq * R_max
        log_Mdyn    = np.log10(max(M_dyn_Msun, 1.0))

        return pd.Series({
            "R_max_kpc":        R_max,
            "V_obs_outer":      V_obs,
            "V_baryon_outer":   np.sqrt(V_baryon_sq),
            "dm_fraction":      dm_fraction,
            "log_Mdyn":         log_Mdyn,
        })

    print("Aggregating radial rotation curves from 'newt-mass.csv'...")
    df_newt_agg = df_newt.groupby("Galaxy_clean", group_keys=False).apply(calc_newt_summary).reset_index()

    # Perform multi-way dataset fusion
    merged = df_galaxies.merge(df_newt_agg, on="Galaxy_clean", how="inner")
    merged = merged.merge(df_stellar[["Galaxy_clean", "logM", "color", "M_L"]], on="Galaxy_clean", how="left")
    merged = merged.merge(df_btf[["Galaxy_clean", "log_Mb", "Vf", "Vmax"]], on="Galaxy_clean", how="left")

    # Feature Engineering & Normalization
    # 1. Log Stellar Mass (M_sun)
    merged["stellar_mass_logM"] = merged["logM"].fillna(np.log10(np.maximum(merged["L_3_6"] * 0.5 * 1e9, 1e-3)))
    
    # 2. HI Gas Mass (10^9 M_sun)
    merged["HI_gas_mass_MHI"] = merged["MHI"]
    
    # 3. Log Baryonic Mass (M_sun)
    calc_Mb = np.log10(np.maximum(10**merged["stellar_mass_logM"] + merged["MHI"] * 1.33 * 1e9, 1.0))
    merged["baryonic_mass_logMb"] = merged["log_Mb"].fillna(calc_Mb)
    
    # 4. Galaxy geometry & kinematics
    merged["effective_radius_Reff"]     = merged["Reff"]
    merged["surface_brightness_SBeff"] = merged["SBeff"]
    merged["disk_scale_length_Rdisk"]  = merged["Rdisk"]
    merged["rotation_velocity_Vflat"]  = merged["Vflat"].replace(0, np.nan).fillna(merged["V_obs_outer"])
    merged["color_3p6_4p5"]            = merged["color"].fillna(merged["color"].median())
    merged["morph_type_T"]             = merged["T"]

    # Select clean final columns
    final_cols = [
        "Galaxy",
        "Galaxy_clean",
        "morph_type_T",
        "D",
        "Inc",
        "stellar_mass_logM",
        "HI_gas_mass_MHI",
        "baryonic_mass_logMb",
        "effective_radius_Reff",
        "surface_brightness_SBeff",
        "disk_scale_length_Rdisk",
        "rotation_velocity_Vflat",
        "color_3p6_4p5",
        "R_max_kpc",
        "V_obs_outer",
        "V_baryon_outer",
        "log_Mdyn",
        "dm_fraction",
    ]
    
    df_combined = merged[final_cols].copy()
    
    # Ensure directory exists and export
    os.makedirs(os.path.dirname(abs_output_csv), exist_ok=True)
    df_combined.to_csv(abs_output_csv, index=False)
    print(f"Successfully exported combined dataset ({len(df_combined)} galaxies) to:\n  '{abs_output_csv}'\n")
    return df_combined

if __name__ == "__main__":
    # Process Train Set
    build_combined_sparc_dataset(
        data_dir="data/train", 
        output_csv="data/sparc_combined_train.csv"
    )
    # Also save to data/train/combined/sparc_combined_train.csv
    build_combined_sparc_dataset(
        data_dir="data/train", 
        output_csv="data/train/combined/sparc_combined_train.csv"
    )
    
    # Process Test Set
    build_combined_sparc_dataset(
        data_dir="data/test", 
        output_csv="data/test/combined/sparc_combined_test.csv"
    )

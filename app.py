import streamlit as st
import pandas as pd
import numpy as np
import io, os, time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

    st.set_page_config(page_title='AI Analytics Assistant', layout='wide')

    # ----------------- Utilities -----------------
    def basic_cleaning(df):
        df = df.copy()
        # trim strings
        for c in df.select_dtypes(include=['object']).columns:
            df[c] = df[c].astype(str).str.strip()
            df[c].replace({'': np.nan, 'NA': np.nan, 'NaN': np.nan}, inplace=True)
        return df

    def build_feature_store(df, target=None, out_dir='out/fe'):
        os.makedirs(out_dir, exist_ok=True)
        df = df.copy()
        if target and target in df.columns:
            y = df[target]
            X = df.drop(columns=[target])
        else:
            y = None; X = df.copy()
        num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
        # simple imputation
        for c in num_cols:
            X[c] = X[c].fillna(X[c].median())
        for c in cat_cols:
            X[c] = X[c].fillna('<<MISSING>>').astype(str)
        # simple encodings & output
        X.to_csv(os.path.join(out_dir, 'features.csv'), index=False)
        meta = {'num_cols': num_cols, 'cat_cols': cat_cols, 'n_rows': int(df.shape[0])}
        with open(os.path.join(out_dir, 'meta.json'),'w') as f:
            import json
            json.dump(meta, f)
        return out_dir

    def run_eda(df, out_dir='out/charts'):
        os.makedirs(out_dir, exist_ok=True)
        charts = []
        # numeric histograms
        num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
        for c in num_cols:
            plt.figure(figsize=(6,3))
            plt.hist(df[c].dropna(), bins=30)
            plt.title(f'Distribution: {c}')
            p = os.path.join(out_dir, f'dist_{c}.png')
            plt.savefig(p, bbox_inches='tight'); plt.close()
            charts.append(p)
        # categorical bars (top values)
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        for c in cat_cols:
            vc = df[c].value_counts().head(20)
            if vc.empty: continue
            plt.figure(figsize=(6,3))
            vc.plot(kind='bar')
            plt.title(f'Categories: {c}')
            p = os.path.join(out_dir, f'bar_{c}.png')
            plt.savefig(p, bbox_inches='tight'); plt.close()
            charts.append(p)
        # correlation heatmap
        if len(num_cols) >= 2:
            plt.figure(figsize=(6,4))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
            p = os.path.join(out_dir, 'correlation.png')
            plt.savefig(p, bbox_inches='tight'); plt.close()
            charts.append(p)
        return charts

    def export_powerbi_files(df, out_dir='out/powerbi'):
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, 'data_for_powerbi.csv')
        df.to_csv(csv_path, index=False)
        # parquet
        try:
            pq_path = os.path.join(out_dir, 'data_for_powerbi.parquet')
            df.to_parquet(pq_path, index=False)
        except Exception:
            pq_path = None
        # mapping
        mapping = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            role = 'Dimension' if dtype.startswith('object') or 'datetime' in dtype else 'Measure'
            mapping[col] = {'dtype': dtype, 'suggested_role': role}
        import json
        with open(os.path.join(out_dir, 'dataset_mapping.json'),'w') as f:
            json.dump(mapping, f, indent=2, default=str)
        with open(os.path.join(out_dir, 'README_powerbi.md'),'w') as f:
            f.write('Power BI: load data_for_powerbi.csv and apply dataset_mapping.json for suggested roles.')
        return {'csv': csv_path, 'parquet': pq_path, 'mapping': os.path.join(out_dir, 'dataset_mapping.json')}

    def generate_m_script(df, filename='dataset.csv'):
        types = []
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                types.append((c,'number'))
            elif pd.api.types.is_datetime64_any_dtype(df[c]):
                types.append((c,'datetime'))
            else:
                types.append((c,'text'))
        m_lines = [f'{{"{c}", type {t}}}' for c,t in types]
        m_script = f"""let
    Source = Csv.Document(File.Contents("{filename}"), [Delimiter=",", Columns={len(df.columns)}, Encoding=65001, QuoteStyle=QuoteStyle.None]),
    PromoteHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    ChangeTypes = Table.TransformColumnTypes(PromoteHeaders, {{{', '.join(m_lines)}}})
in
    ChangeTypes"""
        return m_script

    # ----------------- UI -----------------
    st.title('AI Analytics Assistant — Streamlit')
    st.sidebar.title('Settings')

    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame()

    tab1, tab2, tab3 = st.tabs(['📂 Data & Analysis','📥 Power BI Export','📥 Downloads'])

    with tab1:
        st.header('Upload dataset')
        uploaded = st.file_uploader('Upload CSV', type=['csv'])
        sample_btn = st.button('Use sample dataset')
        if sample_btn and st.session_state.df.empty:
            sample_path = os.path.join(os.path.dirname(__file__),'sample.csv')
            st.session_state.df = pd.read_csv(sample_path)
            st.success('Sample dataset loaded into session.')

        if uploaded is not None:
            st.session_state.df = pd.read_csv(uploaded)
            st.success('Dataset loaded into session.')

        if not st.session_state.df.empty:
            df = basic_cleaning(st.session_state.df)
            st.subheader('Preview')
            st.dataframe(df.head())

            # KPIs
            c1, c2, c3 = st.columns(3)
            c1.metric('Rows', df.shape[0])
            c2.metric('Columns', df.shape[1])
            c3.metric('Missing %', f"{round(df.isna().mean().mean()*100,2)}%")

            # target selection
            default_target = 'churn' if 'churn' in df.columns else df.columns[-1]
            target = st.selectbox('Select target column (optional)', options=['None']+list(df.columns), index=list(df.columns).index(default_target)+1 if default_target in df.columns else 0)
            target = None if target=='None' else target

            if st.button('Run analysis & build feature store'):
                progress = st.progress(0)
                st.info('1/4 — Running EDA...')
                charts = run_eda(df, out_dir=os.path.join('out','charts'))
                progress.progress(25)
                st.info('2/4 — Building feature store...')
                fe_dir = build_feature_store(df, target, out_dir=os.path.join('out','fe'))
                progress.progress(60)
                st.info('3/4 — Exporting Power BI files...')
                bi_paths = export_powerbi_files(df, out_dir=os.path.join('out','powerbi'))
                progress.progress(85)
                st.info('4/4 — Finalizing report...')
                # simple report: save html
                report_path = os.path.join('out','report.html')
                with open(report_path,'w') as f:
                    f.write('<html><body><h1>Auto Report</h1></body></html>')
                progress.progress(100)
                st.success('Analysis complete — switch to Power BI Export tab.')

    with tab2:
        st.header('Power BI Export')
        if st.session_state.df.empty:
            st.info('Run analysis first or upload a CSV in Data & Analysis tab.')
        else:
            df = st.session_state.df
            st.subheader('Preview')
            st.dataframe(df.head())
            # download CSV
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button('⬇️ Download CSV', csv_bytes, file_name='data_for_powerbi.csv', mime='text/csv')
            # download parquet if supported
            try:
                buf = io.BytesIO(); df.to_parquet(buf, index=False); buf.seek(0)
                st.download_button('⬇️ Download Parquet', buf.getvalue(), file_name='data_for_powerbi.parquet', mime='application/octet-stream')
            except Exception:
                st.info('Parquet export not available in this environment.')
            m_script = generate_m_script(df, filename='data_for_powerbi.csv')
            st.subheader('Power Query M script (copy & paste into Power BI)')
            st.code(m_script, language='powerquery-m')
            st.download_button('⬇️ Download M script', m_script.encode('utf-8'), file_name='load_powerbi_m_script.pq', mime='text/plain')

    with tab3:
        st.header('Downloads & Artifacts')
        out_zip = 'streamlit_assistant_final.zip'
        st.write('Available artifacts in `out/` (produced after running analysis)')
        if os.path.exists('out'):
            for root, dirs, files in os.walk('out'):
                for fn in files:
                    st.write('-', os.path.join(root, fn))
        st.write('You can also download the entire project ZIP from the project folder.')

    st.sidebar.markdown('---')
    st.sidebar.write('Built by AI Assistant — enjoy!')

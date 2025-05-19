class PhoneAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        self.display_encoder = LabelEncoder()

        # self.clf = RandomForestClassifier(n_estimators=150, class_weight='balanced')
        # self.reg = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1)
        self.metrics = {}
        self.feature_names = []
        self.y_test_cls = None
        self.y_pred_cls = None
        self.estimator_clasificator = RandomForestClassifier(random_state=42)
        self.Param_Grid_clasification = {
              'n_estimators': [50, 100, 200],
              'max_depth': [None, 10, 20, 30],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'class_weight': [None, 'balanced']
        }
        self.clf = GridSearchCV(estimator=self.estimator_clasificator, param_grid=self.Param_Grid_clasification,
                          cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
        self.X_train_clf = None
        self.X_test_clf = None
        self.y_train_clf = None
        self.y_test_clf = None
        self.y_clf = None
        self.df_data_train = None
        self.new_display_columns_one_hot = None


    def preprocess(self, df):
        # Paso 1: Limpieza básica
        df = df.copy()
        df = df.drop_duplicates(subset='name', keep='first')

        # Paso 2: Limpieza de precios
        df['price'] = df['price'].str.replace(r'[^\d.]', '', regex=True).astype(float)

        df['ratings'] = df['ratings'].str.replace(',','.').astype(float)
        # Paso 7: Normalización de batería
        df['battery'] = df['battery'].str.extract(r'(\d+)', expand=False).astype(float)

        # Paso 3: Manejo de valores nulos
        num_cols = ['ratings', 'internal_storage(GB)', 'storage_ram(GB)', 'battery']

        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())


        # Paso 4: Limpieza de almacenamiento expandible
        df['expandable_storage(TB)'] = df['expandable_storage(TB)'].apply(clean_storage)

        # Paso 5: Extracción de características técnicas
        df[['primary_camera_mp', 'num_cameras']] = df['primary_camera'].apply(extract_camera_specs).apply(pd.Series)
        df[['display_tech', 'refresh_rate']] = df['display'].apply(parse_display).apply(pd.Series)
        df['5G_support'] = df['network'].str.contains('5G', na=False).astype(int)


        # Paso 6: Clasificación de gama
        df['gama'] = df.apply(classify_gama, axis=1)
        df['gama_encoded'] = self.le.fit_transform(df['gama'])
        df['display_tech'] = self.display_encoder.fit_transform(df['display_tech'] )
        return df

    def preprocess_test_data(self, df):

        # Paso 7: Normalización de batería
        df['battery'] = df['battery'].str.extract(r'(\d+)', expand=False).astype(float)

        # Paso 3: Manejo de valores nulos
        num_cols = ['internal_storage(GB)', 'storage_ram(GB)', 'battery']

        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())


        # Paso 4: Limpieza de almacenamiento expandible
        df['expandable_storage(TB)'] = df['expandable_storage(TB)'].apply(clean_storage)

        # Paso 5: Extracción de características técnicas
        df[['primary_camera_mp', 'num_cameras']] = df['primary_camera'].apply(extract_camera_specs).apply(pd.Series)
        df[['display_tech', 'refresh_rate']] = df['display'].apply(parse_display).apply(pd.Series)
        df['5G_support'] = df['network'].str.contains('5G', na=False).astype(int)
        df['display_tech'] = self.display_encoder.fit_transform(df['display_tech'] )


        return df

    def filter_outliers_price(self, df, column):
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5*iqr
        upper_bound = q3 + 1.5*iqr

        # Filtrar dataset para regresión
        return df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

    def train_models(self, df):
        # Selección de características

        # df_reg =df.copy()
        # df_reg = self.filter_outliers_price(df_reg, 'price')

        self.feature_names = [
            'internal_storage(GB)', 'storage_ram(GB)', 'expandable_storage(TB)',
            'primary_camera_mp', 'num_cameras', 'refresh_rate', '5G_support', 'battery','display_tech'
        ]

        #####
        ##Clasificacion
        ####
        self.df_data_train = df[self.feature_names]
        X_clf = self.scaler.fit_transform(self.df_data_train)
        self.y_clf = df['gama_encoded']

        # Entrenamiento
        self.X_train_clf, self.X_test_clf, self.y_train_clf, self.y_test_clf = train_test_split(
        X_clf, self.y_clf, test_size=0.2, random_state=42
        )
        ## Scaler
        scaler = RobustScaler()
        self.X_train_clf = scaler.fit_transform(self.X_train_clf)
        self.X_test_clf = scaler.transform(self.X_test_clf)

        self.clf.fit(self.X_train_clf, self.y_train_clf)

        clf_report = classification_report(self.y_test_clf, self.clf.predict(self.X_test_clf),
                                           target_names=self.le.classes_, output_dict=True)

        self.y_test_cls = self.y_test_clf
        self.y_pred_cls = self.clf.predict(self.X_test_clf)

        #####
        ##Regresion
        ####

        # X_reg = self.scaler.transform(df_reg[self.feature_names])
        # y_reg = df_reg['price']
        # X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        # ## Scaler
        # scaler = RobustScaler()
        # X_train_reg = scaler.fit_transform(X_train_reg)
        # X_test_reg = scaler.transform(X_test_reg)

        # self.reg.fit(X_train_reg, y_train_reg)

        # Métricas

        # reg_metrics = {
        #     'rmse': np.sqrt(mean_squared_error(y_test_reg, self.reg.predict(X_test_reg))),
        #     'r2': r2_score(y_test_reg, self.reg.predict(X_test_reg))
        # }

        self.metrics = {
            'classification': clf_report,
            # 'regression': reg_metrics
        }

    def visualize_coor(self,df):
        sns.set_theme(style="ticks")
        sns.pairplot(df, hue="gama")

    def visualize_data(self, df):
        plt.figure(figsize=(15, 10))

        # # Visualización 1
        plt.subplot(2, 2,1)
        sns.boxplot(x=df['price'])
        plt.title(f"Outliers en price")

        # Visualización 2
        plt.subplot(2,2,2)
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
        plt.title('Matriz de Correlación')

        plt.tight_layout()
        plt.show()

    def evaluate_phone(self, specs):
        phone_df = pd.DataFrame([specs])
        phone_df = self.preprocess_test_data(phone_df)

        try:
            X = self.scaler.transform(phone_df[self.feature_names])
        except KeyError as e:
            raise ValueError(f"Missing feature: {e}")

        # Manejo de nuevas categorías en tiempo real
        try:
            best_rf = self.clf.best_estimator_

            gama = self.le.inverse_transform(best_rf.predict(X))[0]
        except ValueError as e:
            if "unseen labels" in str(e):
                print("Advertencia: Se detectó nueva categoría, usando clasificación alternativa")
                gama = phone_df['gama'].iloc[0]  # Usar la gama generada en el preprocesamiento
            else:
                raise

        # price_pred = self.reg.predict(X)[0]

        # Cálculo seguro de métricas técnicas
        tech_scores = {
            'fotografia': 0.4*(phone_df.get('primary_camera_mp', 0).iloc[0]/200) +
                        0.6*(phone_df.get('num_cameras', 0).iloc[0]/5),
            'gaming': 0.7*(phone_df.get('refresh_rate', 60).iloc[0]/240) +
                    0.3*(phone_df.get('storage_ram(GB)', 0).iloc[0]/16),
            'bateria': phone_df.get('battery', 0).iloc[0]/10000,
            'pantalla': 0.5*(phone_df.get('refresh_rate', 60).iloc[0]/240) +
                    0.5*(phone_df.get('display_tech', 0).iloc[0]/3)}

        return {
            'gama': gama,
            # 'precio_estimado': round(price_pred, 2),
            'puntajes_tecnicos': {k: round(v, 2) for k, v in tech_scores.items()}
        }

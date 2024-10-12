if __name__ == '__main__':
    print("""
        from.sklearn import Model
          
        model=Model()
        
        model.fit(X_train,y_train)
          
        print('Test Accuracy:', model.score(X_test,y_test))

        print(model.predict(x))
          
    """)

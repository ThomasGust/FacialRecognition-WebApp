#from app import db


def get_models(db):
    class jmods(db.Model):

        _id = db.Column("id", db.Integer, primary_key=True)
        created_by = db.Column(db.String)
        mname = db.Column(db.String)
        model = db.Column(db.JSON)
        mdt = db.Column(db.Float)
        mlen = db.Column(db.Integer)

        def __init__(self, created_by, mname, model, mdt):
            self.created_by = created_by
            self.mname = mname
            self.model = model
            self.mdt = mdt
            self.mlen = len(model)

    class users(db.Model):

        _id = db.Column("id", db.Integer, primary_key=True)
        email = db.Column(db.String)
        pwh = db.Column(db.String)

        def __init__(self, email, pwh):
            self.email = email
            self.pwh = pwh
    
    return jmods, users
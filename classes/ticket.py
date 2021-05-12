import functions


class Ticket:
    def __init__(self, ticket_row, object_material):
        self.ticket_date = functions.to_datetime(ticket_row["fecha"])
        self.material = object_material
        # self.status = ticket_row["estado_ot"]
        self.amount = int(ticket_row["cantidad"])
        self.grep = ticket_row["grep"]
        self.set_material()

        self.is_closed = False
        self.days_to_close = None
        self.amount_left = self.amount
        self.is_broken = False
        self.is_partial_closed = False

    # ------------------------------------------------------------------------------------------------------------------

    def __repr__(self):
        return "<GREP>: {} <Material>: {} <Quantity>: {}".format(self.grep, self.material.catalog, self.amount)

    # Setters-----------------------------------------------------------------------------------------------------------
    def set_material(self):
        self.material.set_ticket(self)

def strategy(labels, prices, fee, constant_transaction_fee=True):
    """
    :param fee: if constant_transaction_fee is False, fee is a percent of the traded
    money (fee=0.05 for example); if constant_transaction_fee is True, fee is a constant number of units in certain
    currency (like 1 euro)
    :param constant_transaction_fee: True/False
    """
    i, totalTransactionLength = 0, 0
    buyPoint, sellPoint, gain, totalGain, shareNumber, moneyTemp, maximumMoney = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    money, minimumMoney = 10000.0, 10000.0
    maximumLost = 100.0
    totalPercentProfit, maximumProfitPercent, maximumLostPercent, maximumGain = 0.0, 0.0, 0.0, 0.0
    transactionCount, successTransactionCount, failedTransactionCount = 0, 0, 0
    buyPointBAH, shareNumberBAH, moneyBAH = 10000.0, 10000.0, 10000.0
    forceSell = False

    sharpaR, prof, oneDayProf, numberOfDay, k = 0, 0, 0, 0, 0

    # TODO: why buyPoint and sell Point are multiplied by 100????

    print(f"\n --------- STRATEGY TEST START: fee = {fee}, constant_fee = {constant_transaction_fee} ---------\n")
    print(f"Start Capital: {money}")
    while k < len(labels) - 1:

        # dailyProfit[k] = 0.0

        if labels[k] == 1.0:
            buyPoint = prices[k]
            buyPoint = buyPoint * 100
            if constant_transaction_fee:
                shareNumber = (money - fee) / buyPoint
            else:
                shareNumber = (money - money * fee) / buyPoint
            forceSell = False

            for j in range(k, len(labels) - 1):
                sellPoint = prices[j]
                sellPoint = sellPoint * 100
                if constant_transaction_fee:
                    moneyTemp = (shareNumber * sellPoint) - fee
                else:
                    help = shareNumber * sellPoint
                    moneyTemp = help - help * fee
                # stop loss %10
                # if(money*0.95>moneyTemp){
                # 	money=moneyTemp;
                # 	forceSell=true;
                # }
                if labels[j] == 2.0 or forceSell:
                    sellPoint = prices[j]
                    sellPoint = sellPoint * 100
                    gain = sellPoint - buyPoint
                    if gain > 0:
                        successTransactionCount += 1
                    else:
                        failedTransactionCount += 1
                    if gain >= maximumGain:
                        maximumGain = gain
                        maximumProfitPercent = maximumGain / buyPoint * 100
                    if gain <= maximumLost:
                        maximumLost = gain
                        maximumLostPercent = maximumLost / buyPoint * 100
                    if constant_transaction_fee:
                        moneyTemp = (shareNumber * sellPoint) - fee
                    else:
                        help = shareNumber * sellPoint
                        moneyTemp = help - help * fee
                    money = moneyTemp
                    if money > maximumMoney:
                        maximumMoney = money
                    if money < minimumMoney:
                        minimumMoney = money
                    transactionCount += 1
                    # print(f'{transactionCount}.({k + 1}-{j + 1}) => {round((gain * shareNumber), 2)} Capital: {round(money, 2)}')
                    # prof = round(round((gain * shareNumber), 2) / (money - (gain * shareNumber)), 4)
                    # numberOfDay = j - k
                    # oneDayProf = round((prof / numberOfDay), 4)
                    # for m in range(k + 1, j + 1):
                    #     dailyProfit[m] = oneDayProf

                    totalPercentProfit = totalPercentProfit + (gain / buyPoint)

                    totalTransactionLength = totalTransactionLength + (j - k)
                    k = j + 1
                    totalGain = totalGain + gain
                    break
        k += 1

    # print("dailyProfit[z]")
    # for z in range(0, dailyProfit.length):
    #     print(f'{z}:{dailyProfit[z]}')
    # sharpaR = findSharpaRatio(dailyProfit)
    #
    # print("Sharpa Ratio of Our System=>" + sharpaR)
    print(f"Our System => totalMoney = {round(money, 2)}")

    buyPointBAH = prices[0]
    if constant_transaction_fee:
        shareNumberBAH = (moneyBAH - fee) / buyPointBAH
        moneyBAH = (prices[len(labels) - 1] * shareNumberBAH) - fee
    else:
        shareNumberBAH = (moneyBAH - moneyBAH * fee) / buyPointBAH
        help = prices[len(labels) - 1] * shareNumberBAH
        moneyBAH = help - help * fee

    print(f'BAH => totalMoney = {round(moneyBAH, 2)}')

    numberOfMinutes = len(labels) - 1
    numberOfHours = numberOfMinutes / 60
    numberOfWeeks = numberOfHours/ (24 * 7)

    # print(f"Our System Annualized return % => {round(((math.pow(money / 10000.0, 0.2) - 1) * 100), 2)}%")  # 5 years 0.2
    # print(
    #     f"BaH Annualized return % => {round(((math.exp(math.log(moneyBAH / 10000.0)) - 1) * 100), 2)} %")
    print(f"Number of transaction/2 => {round(transactionCount, 1)} #")
    print(f"Number of transaction => {round(2*transactionCount, 1)} #")

    if transactionCount == 0:
        print(f"Percent success of transaction => 0%")
        print(f"Average percent profit per transaction => 0%")
        print(f"Average transaction length => 0#")
    else:
        print(f"Percent success of transaction => {round((successTransactionCount / transactionCount) * 100, 2)} %")
        print(f"Average percent profit per transaction => {round((totalPercentProfit / transactionCount * 100), 2)} %")
        print(f"Average transaction length => {totalTransactionLength / transactionCount} #")

    print(f"Maximum profit percent in transaction=> {round(maximumProfitPercent, 2)} %")
    print(f"Maximum loss percent in transaction=> {round(maximumLostPercent, 2)} %")
    print(f"Maximum capital value=> {round(maximumMoney, 2)}")
    print(f"Minimum capital value=> {round(minimumMoney, 2)}")
    print("\n--------- STRATEGY TEST END ---------\n")
    # print(f"Idle Ratio %=> {round((len(labels) - totalTransactionLength / len(labels) * 100), 2)} %\n\n")
    return round(money, 2)
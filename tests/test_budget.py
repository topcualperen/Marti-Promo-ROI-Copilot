from model.budget import campaign_stats, compare


def test_smart_beats_random_at_same_budget(small_db):
    r = compare(budget=50_000)
    assert r["akilli_uplift"]["ek_yolculuk"] > r["mevcut_rastgele"]["ek_yolculuk"]
    assert r["akilli_uplift"]["harcanan_tl"] <= 50_000


def test_same_result_needs_less_budget(small_db):
    r = compare()
    assert r["ayni_sonuc_icin_gereken_butce_tl"] < r["butce_tl"]
    assert r["tasarruf_orani"] > 0.3


def test_loyalty_campaign_is_the_least_efficient(small_db):
    stats = campaign_stats()
    worst = stats.sort_values("ek_yolculuk_basi_maliyet_tl").iloc[-1]
    assert worst["kampanya_adi"] == "Sadakat bonusu"
